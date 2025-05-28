"""
Forecast cost doubling rates from processed data.
"""
import argparse
import logging
import os
import pandas as pd
import numpy as np
import prototype_phd.data_utils as data_utils
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple
from .schemas import CostTrend

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast cost doubling rates")
    parser.add_argument("--input", required=True, help="Path to processed data CSV")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    return parser.parse_args()

def fit_cost_trend(df: pd.DataFrame, diff_col: str = "bin_power", cost_col: str = "generation_cost") -> CostTrend:
    """
    Fit cost doubling trend based on difficulty and cost.
    
    Args:
        df: DataFrame containing the data
        diff_col: Column containing difficulty measure
        cost_col: Column containing cost
        
    Returns:
        CostTrend object with doubling rate and intercept
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
        return CostTrend(doubling_rate=1.0, intercept=0.0, r_squared=0.0)
    
    # Fit linear model: log2(cost) = intercept + slope * difficulty
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Slope represents how many difficulty units cause cost to double
    slope = model.coef_[0]
    doubling_rate = 1 / slope if slope != 0 else float('inf')
    
    # Intercept is the base cost (in log2 scale) at difficulty 0
    intercept = model.intercept_
    
    return CostTrend(
        doubling_rate=float(doubling_rate),
        intercept=float(intercept),
        r_squared=float(r_squared)
    )

def fit_weighted_cost_trend(df: pd.DataFrame, diff_col: str = "bin_power", cost_col: str = "generation_cost", 
                           weight_col: str = "weight") -> CostTrend:
    """
    Fit cost doubling trend with weights (e.g., for recency weighting).
    
    Args:
        df: DataFrame containing the data
        diff_col: Column containing difficulty measure
        cost_col: Column containing cost
        weight_col: Column containing weights
        
    Returns:
        CostTrend object with doubling rate and intercept
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
        return CostTrend(doubling_rate=1.0, intercept=0.0, r_squared=0.0)
    
    # Fit linear model: log2(cost) = intercept + slope * difficulty
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Slope represents how many difficulty units cause cost to double
    slope = model.coef_[0]
    doubling_rate = 1 / slope if slope != 0 else float('inf')
    
    # Intercept is the base cost (in log2 scale) at difficulty 0
    intercept = model.intercept_
    
    return CostTrend(
        doubling_rate=float(doubling_rate),
        intercept=float(intercept),
        r_squared=float(r_squared)
    )

def fit_cost_trends_by_model(df: pd.DataFrame) -> Dict[str, CostTrend]:
    """
    Fit cost trends for each model in the dataset.
    
    Args:
        df: DataFrame with processed data
    
    Returns:
        Dictionary mapping model names to CostTrend objects
    """
    results = {}
    
    # Add an aggregate trend across all models
    logging.info("Fitting aggregate cost trend across all models")
    try:
        aggregate_trend = fit_cost_trend(df)
        results["aggregate"] = aggregate_trend.model_dump()
    except Exception as e:
        logging.error(f"Error fitting aggregate cost trend: {e}")
    
    # Fit model-specific trends
    for model, model_df in df.groupby("model"):
        logging.info(f"Fitting cost trend for model: {model}")
        try:
            trend = fit_cost_trend(model_df)
            results[model] = trend.model_dump()
        except Exception as e:
            logging.error(f"Error fitting cost trend for model {model}: {e}")
    
    # Add a recency-weighted trend if dates are available
    if "date" in df.columns and not df["date"].isna().all():
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
            weighted_trend = fit_weighted_cost_trend(df)
            results["recency_weighted"] = weighted_trend.model_dump()
        except Exception as e:
            logging.error(f"Error fitting recency-weighted cost trend: {e}")
    
    return results

def save_cost_trends(trends: Dict[str, Dict[str, float]], output_path: str):
    """Save cost trends to CSV file."""
    # Convert to DataFrame
    records = []
    for model, trend in trends.items():
        record = trend.copy()
        record["model"] = model
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

def generate_ensemble_forecast(trends: Dict[str, Dict[str, float]], 
                              difficulty_range: List[float]) -> pd.DataFrame:
    """
    Generate an ensemble forecast of costs across a range of difficulties.
    
    Args:
        trends: Dictionary of cost trend parameters by model/approach
        difficulty_range: List of difficulty values to forecast for
    
    Returns:
        DataFrame with forecasted costs by model and difficulty
    """
    forecasts = []
    
    for model, trend in trends.items():
        doubling_rate = trend.get('doubling_rate', 1.0)
        intercept = trend.get('intercept', 0.0)
        
        for difficulty in difficulty_range:
            # Calculate forecasted cost at this difficulty
            log2_cost = intercept + (difficulty / doubling_rate)
            cost = 2.0 ** log2_cost
            
            forecasts.append({
                'model': model,
                'difficulty': difficulty,
                'forecasted_cost': cost,
                'log2_cost': log2_cost,
                'doubling_rate': doubling_rate,
                'base_cost': 2.0 ** intercept
            })
    
    return pd.DataFrame(forecasts)

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Loading processed data from {args.input}")
    df = pd.read_csv(args.input)
    logging.info(f"Loaded {len(df)} records")
    
    logging.info("Fitting cost trends by model and aggregate approaches")
    trends = fit_cost_trends_by_model(df)
    logging.info(f"Fitted trends for {len(trends)} approaches")
    
    # Generate ensemble forecasts
    difficulty_range = np.arange(0, 20, 0.5)  # Forecast for difficulties 0-20
    forecasts_df = generate_ensemble_forecast(trends, difficulty_range)
    
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
