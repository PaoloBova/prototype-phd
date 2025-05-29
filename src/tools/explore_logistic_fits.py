"""
Utilities for exploring and visualizing logistic curve fits on model performance data.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explore logistic curve fits")
    parser.add_argument("--fits", required=True, help="Path to logistic fit results JSON file")
    parser.add_argument("--data", required=False, help="Optional path to processed data CSV for validation plots")
    parser.add_argument("--output", default="reports/logistic_fit_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--format", default="png", help="Output format (png, pdf, svg)")
    return parser.parse_args()

def logistic_function(x: np.ndarray, threshold: float, slope: float) -> np.ndarray:
    """Calculate logistic function values."""
    return 1.0 / (1.0 + np.exp(-slope * (x - threshold)))

def load_logistic_fits(fits_path: str) -> Dict:
    """
    Load logistic fit parameters from JSON file.
    
    Args:
        fits_path: Path to logistic fit results JSON
    
    Returns:
        Dictionary of fit parameters by model
    """
    with open(fits_path, 'r') as f:
        fits = json.load(f)
    
    # Convert dates from strings to datetime objects
    for model_params in fits.values():
        if 'date' in model_params:
            model_params['date'] = datetime.fromisoformat(model_params['date'].split('+')[0])
    
    return fits

def plot_model_curves(fits: Dict, output_dir: str, fmt: str = "png"):
    """
    Plot fitted logistic curves for each model.
    
    Args:
        fits: Dictionary of fit parameters by model
        output_dir: Directory to save plots
        fmt: File format for output
    """
    plt.figure(figsize=(12, 8))
    
    # Find the range of thresholds to determine x-axis limits
    thresholds = [params['threshold'] for params in fits.values()]
    min_threshold = min(thresholds) - 3
    max_threshold = max(thresholds) + 3
    
    # Create difficulty range for curves
    x = np.linspace(min_threshold, max_threshold, 1000)
    
    # Sort models by threshold for better coloring and legend ordering
    sorted_models = sorted(fits.keys(), key=lambda m: fits[m]['threshold'])
    
    # Plot each model's curve
    for model in sorted_models:
        params = fits[model]
        threshold = params['threshold']
        slope = params['slope']
        
        # Calculate success probabilities
        y = logistic_function(x, threshold, slope)
        
        # Plot the curve
        plt.plot(x, y, label=f"{model} (T={threshold:.2f}, S={slope:.2f})")
        
        # Mark threshold point (where success probability = 0.5)
        plt.scatter([threshold], [0.5], marker='o')
    
    # Add reference line at 0.5 probability
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel("Task Difficulty")
    plt.ylabel("Success Probability")
    plt.title("Logistic Curves by Model")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"logistic_curves.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved logistic curves plot to {output_path}")

def plot_model_parameters(fits: Dict, output_dir: str, fmt: str = "png"):
    """
    Plot threshold vs slope parameters for all models.
    
    Args:
        fits: Dictionary of fit parameters by model
        output_dir: Directory to save plots
        fmt: File format for output
    """
    plt.figure(figsize=(10, 6))
    
    thresholds = []
    slopes = []
    model_labels = []
    
    # Extract parameters
    for model, params in fits.items():
        thresholds.append(params['threshold'])
        slopes.append(params['slope'])
        model_labels.append(model)
    
    # Create scatter plot
    sc = plt.scatter(thresholds, slopes, s=80, alpha=0.7)
    
    # Add model labels as annotations
    for i, model in enumerate(model_labels):
        plt.annotate(model, (thresholds[i], slopes[i]), 
                    xytext=(5, 5), textcoords="offset points")
    
    plt.xlabel("Threshold Parameter")
    plt.ylabel("Slope Parameter")
    plt.title("Model Parameter Comparison")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"model_parameters.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved model parameters plot to {output_path}")

def plot_timeline(fits: Dict, output_dir: str, fmt: str = "png"):
    """
    Plot threshold parameters over time.
    
    Args:
        fits: Dictionary of fit parameters by model
        output_dir: Directory to save plot
        fmt: File format for output
    """
    # Filter out entries without dates
    dated_fits = {model: params for model, params in fits.items() if 'date' in params}
    
    if not dated_fits:
        print("No models with dates found, skipping timeline plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Extract dates and thresholds
    models = []
    dates = []
    thresholds = []
    slopes = []
    
    for model, params in dated_fits.items():
        models.append(model)
        dates.append(params['date'])
        thresholds.append(params['threshold'])
        slopes.append(params['slope'])
    
    # Sort by date
    sorted_indices = np.argsort(dates)
    dates = [dates[i] for i in sorted_indices]
    thresholds = [thresholds[i] for i in sorted_indices]
    models = [models[i] for i in sorted_indices]
    slopes = [slopes[i] for i in sorted_indices]
    
    # Plot thresholds over time
    plt.subplot(2, 1, 1)
    plt.plot(dates, thresholds, marker='o', linestyle='-')
    for i, model in enumerate(models):
        plt.annotate(model, (dates[i], thresholds[i]), 
                   xytext=(5, 0), textcoords="offset points", fontsize=8)
    
    plt.ylabel("Threshold Parameter")
    plt.title("Model Capability Timeline")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot slopes over time
    plt.subplot(2, 1, 2)
    plt.plot(dates, slopes, marker='s', linestyle='-', color='orange')
    for i, model in enumerate(models):
        plt.annotate(model, (dates[i], slopes[i]), 
                   xytext=(5, 0), textcoords="offset points", fontsize=8)
    
    plt.ylabel("Slope Parameter")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"capability_timeline.{fmt}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved capability timeline to {output_path}")

def validate_fits_with_data(fits: Dict, data_df: pd.DataFrame, output_dir: str, fmt: str = "png"):
    """
    Create validation plots comparing fitted curves with actual data points.
    
    Args:
        fits: Dictionary of fit parameters by model
        data_df: DataFrame with processed data for validation
        output_dir: Directory to save plots
        fmt: File format for output
    """
    if data_df is None or len(data_df) == 0:
        print("No validation data provided, skipping fit validation plots")
        return
    
    # Create validation directory
    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    # Calculate success rates by difficulty bin for each model
    for model, params in fits.items():
        # Filter data for this model
        model_data = data_df[data_df["alias"] == model]
        
        if len(model_data) == 0:
            print(f"No data found for model {model}, skipping validation plot")
            continue
        
        # Group by bin power and calculate success rate
        grouped = model_data.groupby("bin_power")["score_binarized"].agg(
            success_rate="mean", 
            count="count"
        ).reset_index()
        
        if len(grouped) < 3:
            print(f"Insufficient data points for model {model}, skipping validation plot")
            continue
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot actual data points
        plt.scatter(grouped["bin_power"], grouped["success_rate"], 
                   s=grouped["count"] * 2, alpha=0.7, label="Actual Data")
        
        # Plot fitted curve
        x = np.linspace(min(grouped["bin_power"]) - 1, max(grouped["bin_power"]) + 1, 1000)
        y = logistic_function(x, params["threshold"], params["slope"])
        plt.plot(x, y, 'r-', label="Fitted Curve")
        
        # Highlight the threshold
        plt.axvline(params["threshold"], color='r', linestyle='--', 
                   alpha=0.7, label=f"Threshold = {params['threshold']:.2f}")
        
        plt.xlabel("Task Difficulty (bin)")
        plt.ylabel("Success Rate")
        plt.title(f"Logistic Fit Validation for {model}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # Save figure
        safe_model_name = model.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(validation_dir, f"validation_{safe_model_name}.{fmt}")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved validation plot for {model} to {output_path}")

def create_summary_stats(fits: Dict, output_dir: str):
    """
    Create a summary table of fit parameters.
    
    Args:
        fits: Dictionary of fit parameters by model
        output_dir: Directory to save summary
    """
    records = []
    
    for model, params in fits.items():
        record = {
            "model": model,
            "threshold": params["threshold"],
            "slope": params["slope"]
        }
        
        if "date" in params:
            record["date"] = params["date"]
            
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Sort by threshold
    df = df.sort_values("threshold")
    
    # Calculate basic statistics
    stats = {
        "mean_threshold": np.mean(df["threshold"]),
        "std_threshold": np.std(df["threshold"]),
        "min_threshold": np.min(df["threshold"]),
        "max_threshold": np.max(df["threshold"]),
        "mean_slope": np.mean(df["slope"]),
        "std_slope": np.std(df["slope"]),
        "min_slope": np.min(df["slope"]),
        "max_slope": np.max(df["slope"]),
        "model_count": len(df)
    }
    
    # Save detailed fits to CSV
    detail_path = os.path.join(output_dir, "logistic_fit_details.csv")
    df.to_csv(detail_path, index=False)
    
    # Save summary stats to CSV
    summary_path = os.path.join(output_dir, "logistic_fit_summary.csv")
    pd.DataFrame([stats]).to_csv(summary_path, index=False)
    
    print(f"Saved detailed fit parameters to {detail_path}")
    print(f"Saved summary statistics to {summary_path}")
    
    # Print summary to console
    print("\nFit Summary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Loading logistic fit results from {args.fits}")
    fits = load_logistic_fits(args.fits)
    print(f"Loaded fit parameters for {len(fits)} models")
    
    # Load optional validation data
    data_df = None
    if args.data:
        print(f"Loading processed data from {args.data}")
        data_df = pd.read_csv(args.data)
        print(f"Loaded {len(data_df)} data records for validation")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate visualizations
    plot_model_curves(fits, args.output, args.format)
    plot_model_parameters(fits, args.output, args.format)
    plot_timeline(fits, args.output, args.format)
    create_summary_stats(fits, args.output)
    
    # Generate validation plots if data is available
    validate_fits_with_data(fits, data_df, args.output, args.format)
    
    print(f"All visualizations saved to {args.output}")

if __name__ == "__main__":
    main()
