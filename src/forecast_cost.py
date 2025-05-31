"""
Forecast cost doubling rates from processed data.
"""
import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import prototype_phd.data_utils as data_utils
from sklearn.linear_model import LinearRegression
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from pydantic import BaseModel, Field
from .schemas import CostTrend
from .cost_trend_utils import aggregate_doubling_rates, analyze_doubling_rates, calculate_weighted_doubling_rate

class CostForecastConfig(BaseModel):
    """Configuration for cost forecasts."""
    aggregation_methods: List[str] = Field(
        ["geometric_mean", "median"], 
        description="Methods to use for aggregating doubling rates"
    )
    weighting_methods: List[str] = Field(
        ["uniform", "r_squared"], 
        description="Methods to use for weighting doubling rates"
    )
    confidence_level: float = Field(
        0.95, 
        description="Confidence level for intervals (0-1)"
    )
    include_model_specific: bool = Field(
        True, 
        description="Whether to include model-specific trends"
    )
    include_aggregate: bool = Field(
        True, 
        description="Whether to include aggregate trends"
    )
    include_recency_weighted: bool = Field(
        True, 
        description="Whether to include recency-weighted trends"
    )
    difficulty_min: float = Field(
        0, 
        description="Minimum difficulty value for forecasts"
    )
    difficulty_max: float = Field(
        20, 
        description="Maximum difficulty value for forecasts"
    )
    difficulty_step: float = Field(
        0.5, 
        description="Step size for difficulty values in forecasts"
    )
    
    class Config:
        arbitrary_types_allowed = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast cost doubling rates")
    parser.add_argument("--input", required=True, help="Path to processed data CSV")
    parser.add_argument("--config", required=False, help="Path to forecast config JSON")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    return parser.parse_args()

def fit_cost_trend(df: pd.DataFrame, diff_col: str = "bin_power", cost_col: str = "generation_cost") -> Tuple[CostTrend, Dict[str, Any]]:
    """
    Fit cost doubling trend based on difficulty and cost, with confidence intervals.
    
    Args:
        df: DataFrame containing the data
        diff_col: Column containing difficulty measure
        cost_col: Column containing cost
        
    Returns:
        CostTrend object with doubling rate and intercept, and confidence data
    """
    # Group by difficulty bin and compute mean cost
    grouped = df.groupby(diff_col)[cost_col].mean().reset_index()
    
    # Convert to log2 scale for linear fitting
    grouped["log2_cost"] = np.log2(grouped[cost_col])
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(grouped["log2_cost"]) & np.isfinite(grouped[diff_col])
    X = grouped.loc[valid_mask, diff_col].values.reshape(-1, 1)
    y = grouped.loc[valid_mask, "log2_cost"].values
    
    if len(X) < 2:
        logging.warning("Not enough valid data points to fit cost trend")
        return CostTrend(doubling_rate=1.0, intercept=0.0, r_squared=0.0), {
            "confidence_intervals": None,
            "raw_data": None
        }
    
    # Fit linear model: log2(cost) = intercept + slope * difficulty
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate standard errors and confidence intervals
    n = len(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Only calculate CIs if we have enough data points
    confidence_data = None
    if n > 2:
        # Calculate standard errors
        df_residual = n - 2
        mse = ss_res / df_residual
        se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X)) ** 2))
        
        # t-statistic for confidence level
        t_value = stats.t.ppf(0.975, df_residual)  # 95% CI
        
        # Confidence interval for slope
        slope_ci_lower = slope - t_value * se_slope
        slope_ci_upper = slope + t_value * se_slope
        
        # Convert to doubling rate CI (invert as doubling_rate = 1/slope)
        # Handle case where CI might include zero
        if slope_ci_lower <= 0 and slope_ci_upper > 0:
            dr_ci_lower = float('inf')
            dr_ci_upper = 1 / slope_ci_upper if slope_ci_upper != 0 else float('inf')
        elif slope_ci_lower > 0 and slope_ci_upper > 0:
            dr_ci_lower = 1 / slope_ci_upper
            dr_ci_upper = 1 / slope_ci_lower
        elif slope_ci_lower < 0 and slope_ci_upper < 0:
            dr_ci_lower = 1 / slope_ci_upper
            dr_ci_upper = 1 / slope_ci_lower
        else:
            dr_ci_lower = float('nan')
            dr_ci_upper = float('nan')
            
        confidence_data = {
            "slope_ci_lower": float(slope_ci_lower),
            "slope_ci_upper": float(slope_ci_upper),
            "doubling_rate_ci_lower": float(dr_ci_lower),
            "doubling_rate_ci_upper": float(dr_ci_upper),
            "df_residual": df_residual,
            "mse": float(mse)
        }
    
    # Slope represents how many difficulty units cause cost to double
    doubling_rate = 1 / slope if slope != 0 else float('inf')
    
    # Return CostTrend object and additional confidence interval data
    trend = CostTrend(
        doubling_rate=float(doubling_rate),
        intercept=float(intercept),
        r_squared=float(r_squared)
    )
    
    # Also return raw data for later analysis
    additional_data = {
        "confidence_intervals": confidence_data,
        "raw_data": {
            "X": X.flatten().tolist(),
            "y": y.tolist(),
            "y_pred": y_pred.tolist()
        }
    }
    
    return trend, additional_data

def fit_weighted_cost_trend(df: pd.DataFrame, diff_col: str = "bin_power", cost_col: str = "generation_cost", 
                           weight_col: str = "weight") -> Tuple[CostTrend, Dict[str, Any]]:
    """
    Fit cost doubling trend with weights (e.g., for recency weighting).
    
    Args:
        df: DataFrame containing the data
        diff_col: Column containing difficulty measure
        cost_col: Column containing cost
        weight_col: Column containing weights
        
    Returns:
        CostTrend object with doubling rate and intercept, and confidence data
    """
    # Group by difficulty bin and compute weighted mean cost
    grouped = df.groupby(diff_col)[[cost_col, weight_col]].apply(
        lambda x: np.average(x[cost_col], weights=x[weight_col])
    ).reset_index(name="weighted_cost")
    
    # Convert to log2 scale for linear fitting
    grouped["log2_cost"] = np.log2(grouped["weighted_cost"])
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(grouped["log2_cost"]) & np.isfinite(grouped[diff_col])
    X = grouped.loc[valid_mask, diff_col].values.reshape(-1, 1)
    y = grouped.loc[valid_mask, "log2_cost"].values
    
    if len(X) < 2:
        logging.warning("Not enough valid data points to fit weighted cost trend")
        return CostTrend(doubling_rate=1.0, intercept=0.0, r_squared=0.0), {
            "confidence_intervals": None,
            "raw_data": None
        }
    
    # Fit linear model: log2(cost) = intercept + slope * difficulty
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate standard errors and confidence intervals
    n = len(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Only calculate CIs if we have enough data points
    confidence_data = None
    if n > 2:
        # Calculate standard errors
        df_residual = n - 2
        mse = ss_res / df_residual
        se_slope = np.sqrt(mse / np.sum((X.flatten() - np.mean(X)) ** 2))
        
        # t-statistic for confidence level
        t_value = stats.t.ppf(0.975, df_residual)  # 95% CI
        
        # Confidence interval for slope
        slope_ci_lower = slope - t_value * se_slope
        slope_ci_upper = slope + t_value * se_slope
        
        # Convert to doubling rate CI (invert as doubling_rate = 1/slope)
        # Handle case where CI might include zero
        if slope_ci_lower <= 0 and slope_ci_upper > 0:
            dr_ci_lower = float('inf')
            dr_ci_upper = 1 / slope_ci_upper if slope_ci_upper != 0 else float('inf')
        elif slope_ci_lower > 0 and slope_ci_upper > 0:
            dr_ci_lower = 1 / slope_ci_upper
            dr_ci_upper = 1 / slope_ci_lower
        elif slope_ci_lower < 0 and slope_ci_upper < 0:
            dr_ci_lower = 1 / slope_ci_upper
            dr_ci_upper = 1 / slope_ci_lower
        else:
            dr_ci_lower = float('nan')
            dr_ci_upper = float('nan')
            
        confidence_data = {
            "slope_ci_lower": float(slope_ci_lower),
            "slope_ci_upper": float(slope_ci_upper),
            "doubling_rate_ci_lower": float(dr_ci_lower),
            "doubling_rate_ci_upper": float(dr_ci_upper),
            "df_residual": df_residual,
            "mse": float(mse)
        }
    
    # Slope represents how many difficulty units cause cost to double
    doubling_rate = 1 / slope if slope != 0 else float('inf')
    
    # Return CostTrend object and additional confidence interval data
    trend = CostTrend(
        doubling_rate=float(doubling_rate),
        intercept=float(intercept),
        r_squared=float(r_squared)
    )
    
    # Also return raw data for later analysis
    additional_data = {
        "confidence_intervals": confidence_data,
        "raw_data": {
            "X": X.flatten().tolist(),
            "y": y.tolist(),
            "y_pred": y_pred.tolist()
        }
    }
    
    return trend, additional_data

def calculate_bootstrap_confidence_intervals(
    values: List[float], 
    statistic=np.median, 
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,  # Increased from 5000
    random_state: Optional[int] = None
) -> Optional[Tuple[float, float]]:
    """
    Calculate confidence intervals using bootstrap resampling with BCa correction.
    
    Args:
        values: List of values to calculate statistic on
        statistic: Function to compute the statistic (e.g., np.median)
        confidence_level: Confidence level for intervals (0-1)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (lower_bound, upper_bound) or None if calculation fails
    """
    if len(values) < 3:
        return None
    
    # Convert to numpy array if not already
    values = np.array(values)
    n = len(values)
    
    # For very small samples, use a more conservative approach
    if n < 10:
        # Use a wider interval for small samples
        if statistic == np.median:
            # For median with small samples, use a distribution-free method
            # Order statistics approach for median CI
            values_sorted = np.sort(values)
            alpha = 1.0 - confidence_level
            
            # Calculate indices for confidence limits
            # This is a simple approximate method for small samples
            lower_idx = max(0, int(n * alpha / 2))
            upper_idx = min(n - 1, int(n * (1 - alpha / 2)))
            
            return float(values_sorted[lower_idx]), float(values_sorted[upper_idx])
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
        
    try:
        # Original statistic value
        theta_hat = statistic(values)
        
        # Generate bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample = np.random.choice(values, size=n, replace=True)
            # Calculate statistic on this sample
            bootstrap_stats.append(statistic(sample))
        
        # Calculate jackknife values for acceleration factor
        jackknife_stats = []
        for i in range(n):
            # Leave one out
            sample = np.concatenate([values[:i], values[i+1:]])
            jackknife_stats.append(statistic(sample))
        
        # Calculate bias-correction factor
        z0 = stats.norm.ppf((np.sum(np.array(bootstrap_stats) < theta_hat) / n_bootstrap))
        
        # Calculate acceleration factor
        jackknife_mean = np.mean(jackknife_stats)
        num = np.sum((jackknife_mean - jackknife_stats) ** 3)
        den = 6.0 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        # Avoid division by zero
        if abs(den) < 1e-10:
            a = 0.0
        else:
            a = num / den
        
        # Calculate BCa confidence interval
        alpha = 1.0 - confidence_level
        alpha_points = [alpha / 2, 1.0 - alpha / 2]
        
        z_values = []
        for alpha_point in alpha_points:
            z_alpha = stats.norm.ppf(alpha_point)
            z_values.append(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)))
        
        # Convert to percentiles
        percentiles = [stats.norm.cdf(z) * 100 for z in z_values]
        
        # Get confidence bounds
        lower_bound = np.percentile(bootstrap_stats, percentiles[0])
        upper_bound = np.percentile(bootstrap_stats, percentiles[1])
        
        # If BCa fails (can happen with small samples), fall back to percentile bootstrap
        if not (np.isfinite(lower_bound) and np.isfinite(upper_bound)):
            alpha = 1.0 - confidence_level
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = np.percentile(bootstrap_stats, lower_percentile)
            upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return float(lower_bound), float(upper_bound)
    except Exception:
        # If BCa bootstrap fails, fall back to simple percentile bootstrap
        try:
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=n, replace=True)
                bootstrap_stats.append(statistic(sample))
            
            # Calculate simple percentile interval
            alpha = 1.0 - confidence_level
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            lower_bound = np.percentile(bootstrap_stats, lower_percentile)
            upper_bound = np.percentile(bootstrap_stats, upper_percentile)
            
            return float(lower_bound), float(upper_bound)
        except:
            return None

def fit_cost_trends_by_model(df: pd.DataFrame, config: CostForecastConfig) -> Dict[str, Dict[str, Any]]:
    """
    Fit cost trends for each model in the dataset based on configuration.
    
    Args:
        df: DataFrame with processed data
        config: Forecast configuration
    
    Returns:
        Dictionary mapping model/approach names to trend data
    """
    results = {}
    additional_data = {}
    
    # Add an aggregate trend across all models if requested
    if config.include_aggregate:
        logging.info("Fitting aggregate cost trend across all models")
        try:
            aggregate_trend, agg_data = fit_cost_trend(df)
            results["aggregate"] = aggregate_trend.model_dump()
            additional_data["aggregate"] = agg_data
        except Exception as e:
            logging.error(f"Error fitting aggregate cost trend: {e}")
    
    # Fit model-specific trends if requested
    if config.include_model_specific:
        for model, model_df in df.groupby("alias"):
            logging.info(f"Fitting cost trend for model: {model}")
            try:
                trend, trend_data = fit_cost_trend(model_df)
                results[model] = trend.model_dump()
                additional_data[model] = trend_data
            except Exception as e:
                logging.error(f"Error fitting cost trend for model {model}: {e}")
    
    # Add a recency-weighted trend if dates are available and requested
    if config.include_recency_weighted and "date" in df.columns and not df["date"].isna().all():
        logging.info("Fitting recency-weighted cost trend")
        try:
            # Convert dates to timestamps for weighting
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            max_date = df["date"].max()
            # Weight by recency (more recent models get higher weights)
            df["weight"] = (df["date"] - df["date"].min()).dt.total_seconds() / \
                           (max_date - df["date"].min()).dt.total_seconds()
            df["weight"] = df["weight"].fillna(0.5)  # Default weight for entries with no date
            
            # Fit weighted trend
            weighted_trend, weighted_data = fit_weighted_cost_trend(df)
            results["recency_weighted"] = weighted_trend.model_dump()
            additional_data["recency_weighted"] = weighted_data
        except Exception as e:
            logging.error(f"Error fitting recency-weighted cost trend: {e}")
    
    # Generate aggregate trends using different methods if there are multiple models
    if len([k for k in results.keys() if k not in ["aggregate", "recency_weighted"]]) > 1:
        # Extract doubling rates from individual models
        doubling_rates = [
            results[model]["doubling_rate"] 
            for model in results
            if model not in ["aggregate", "recency_weighted"] and np.isfinite(results[model]["doubling_rate"])
        ]
        
        if doubling_rates:
            # Create trends with different aggregation methods
            for method in config.aggregation_methods:
                try:
                    # Get aggregated doubling rate
                    agg_rate = aggregate_doubling_rates(doubling_rates, method=method)
                    
                    # Use the average intercept from individual models
                    avg_intercept = np.mean([
                        results[model]["intercept"]
                        for model in results
                        if model not in ["aggregate", "recency_weighted"]
                    ])
                    
                    # Calculate R² as average of individual models
                    avg_r_squared = np.mean([
                        results[model]["r_squared"]
                        for model in results
                        if model not in ["aggregate", "recency_weighted"]
                    ])
                    
                    # Create aggregated trend
                    agg_trend = CostTrend(
                        doubling_rate=float(agg_rate),
                        intercept=float(avg_intercept),
                        r_squared=float(avg_r_squared)
                    )
                    
                    # Analyze doubling rates to get confidence intervals
                    analysis = analyze_doubling_rates(
                        doubling_rates,
                        confidence_level=config.confidence_level
                    )
                    
                    # Save trend and analysis
                    results[f"agg_{method}"] = agg_trend.model_dump()
                    additional_data[f"agg_{method}"] = {
                        "analysis": analysis,
                        "individual_rates": doubling_rates
                    }
                    
                    # Add the appropriate confidence intervals based on aggregation method
                    if method == "geometric_mean" and "geom_mean_ci_lower" in analysis and "geom_mean_ci_upper" in analysis:
                        # Use the geometric mean confidence intervals from analysis
                        results[f"agg_{method}"]["doubling_rate_ci_lower"] = analysis["geom_mean_ci_lower"]
                        results[f"agg_{method}"]["doubling_rate_ci_upper"] = analysis["geom_mean_ci_upper"]
                    
                    elif method == "median" or method == "trimmed_mean":
                        # Calculate bootstrap-based confidence intervals
                        bootstrap_cis = calculate_bootstrap_confidence_intervals(
                            doubling_rates,
                            statistic=np.median if method == "median" else lambda x: stats.trim_mean(x, 0.1),
                            confidence_level=config.confidence_level,
                            random_state=42  # Use fixed seed for reproducibility
                        )
                        
                        if bootstrap_cis:
                            results[f"agg_{method}"]["doubling_rate_ci_lower"] = bootstrap_cis[0]
                            results[f"agg_{method}"]["doubling_rate_ci_upper"] = bootstrap_cis[1]
                    
                except Exception as e:
                    logging.error(f"Error creating {method} aggregated trend: {e}")
    
    # Return results with trends and additional data
    final_results = {}
    for model, trend in results.items():
        final_results[model] = {
            **trend,
            "additional_data": additional_data.get(model, {})
        }
        
        # Add confidence intervals to the main results if available
        ci_data = additional_data.get(model, {}).get("confidence_intervals")
        if ci_data:
            final_results[model]["doubling_rate_ci_lower"] = ci_data["doubling_rate_ci_lower"]
            final_results[model]["doubling_rate_ci_upper"] = ci_data["doubling_rate_ci_upper"]
    
    return final_results

def save_cost_trends(trends: Dict[str, Dict[str, Any]], output_path: str):
    """Save cost trends to CSV file."""
    # Convert to DataFrame
    records = []
    for model, trend in trends.items():
        record = {k: v for k, v in trend.items() if k != "additional_data"}
        record["model"] = model
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    # Save detailed data to JSON
    detailed_path = output_path.replace('.csv', '_detailed.json')
    with open(detailed_path, 'w') as f:
        json.dump(trends, f, indent=2, default=lambda x: str(x) if isinstance(x, (np.float32, np.float64)) else x)

def generate_ensemble_forecast(trends: Dict[str, Dict[str, Any]], config: CostForecastConfig) -> pd.DataFrame:
    """
    Generate an ensemble forecast of costs across a range of difficulties.
    
    Args:
        trends: Dictionary of cost trend parameters by model/approach
        config: Forecast configuration
    
    Returns:
        DataFrame with forecasted costs by model and difficulty
    """
    # Generate difficulty range based on config
    difficulty_range = np.arange(
        config.difficulty_min,
        config.difficulty_max + config.difficulty_step,  # Include max value
        config.difficulty_step
    )
    
    forecasts = []
    
    for model, trend in trends.items():
        doubling_rate = trend.get('doubling_rate', 1.0)
        intercept = trend.get('intercept', 0.0)
        
        # Get confidence intervals if available
        dr_ci_lower = trend.get('doubling_rate_ci_lower', None)
        dr_ci_upper = trend.get('doubling_rate_ci_upper', None)
        
        for difficulty in difficulty_range:
            # Calculate forecasted cost at this difficulty
            log2_cost = intercept + (difficulty / doubling_rate)
            cost = 2.0 ** log2_cost
            
            forecast = {
                'model': model,
                'difficulty': difficulty,
                'forecasted_cost': cost,
                'log2_cost': log2_cost,
                'doubling_rate': doubling_rate,
                'base_cost': 2.0 ** intercept
            }
            
            # Add confidence interval forecasts if available
            if dr_ci_lower is not None and np.isfinite(dr_ci_lower):
                log2_cost_upper = intercept + (difficulty / dr_ci_lower)
                forecast['cost_ci_lower'] = 2.0 ** log2_cost_upper
                
            if dr_ci_upper is not None and np.isfinite(dr_ci_upper):
                log2_cost_lower = intercept + (difficulty / dr_ci_upper)
                forecast['cost_ci_upper'] = 2.0 ** log2_cost_lower
                
            forecasts.append(forecast)
    
    return pd.DataFrame(forecasts)

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Loading processed data from {args.input}")
    df = pd.read_csv(args.input)
    logging.info(f"Loaded {len(df)} records")
    
    # Load configuration if provided, otherwise use defaults
    config = CostForecastConfig()
    if args.config:
        logging.info(f"Loading config from {args.config}")
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            config = CostForecastConfig(**config_data)
        except Exception as e:
            logging.error(f"Error loading config, using defaults: {e}")
    
    logging.info("Fitting cost trends based on configuration")
    trends = fit_cost_trends_by_model(df, config)
    logging.info(f"Fitted trends for {len(trends)} approaches")
    
    # Generate ensemble forecasts
    forecasts_df = generate_ensemble_forecast(trends, config)
    
    logging.info(f"Saving cost trends to {args.out}")
    save_cost_trends(trends, args.out)
    
    # Save forecasts to a separate file
    forecast_path = os.path.join(os.path.dirname(args.out), "cost_forecasts.csv")
    logging.info(f"Saving cost forecasts to {forecast_path}")
    os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
    forecasts_df.to_csv(forecast_path, index=False)
    
    logging.info("Complete")

if __name__ == "__main__":
    main()
