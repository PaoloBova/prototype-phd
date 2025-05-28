"""
Fit logistic curves to processed data.
"""
import argparse
import logging
import os
import pandas as pd
import prototype_phd.data_utils as data_utils
import prototype_phd.stats as stats
from typing import Dict
from .schemas import LogisticFitParams

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit logistic curves to processed data")
    parser.add_argument("--input", required=True, help="Path to processed data directory")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    return parser.parse_args()

def fit_curves_by_model(df: pd.DataFrame) -> Dict[str, LogisticFitParams]:
    """
    Fit logistic curves for each model in the dataset.
    
    Args:
        df: DataFrame with processed data
    
    Returns:
        Dictionary mapping model names to LogisticFitParams objects
    """
    results = {}
    
    models_date_of_release = df.groupby("alias")["date"].min().to_dict()
    
    for model, model_df in df.groupby("alias"):
        logging.info(f"Fitting curve for model: {model}")
        x_cols = ["log2_human_seconds"]
        y_col = "score_binarized"
        logreg_config = stats.LogRegConfig(
            engine="scikit-learn",
            solver="lbfgs",
            C=1,
            max_iter=1000,
        )
        X = model_df[x_cols]
        y = model_df[y_col]
        logreg_result = stats.fit_logistic(X, y, logreg_config)
        coeffs = logreg_result.coeffs
        intercept, slope = coeffs[0], coeffs[1]
        threshold = -intercept / slope
        # Create params object
        params = LogisticFitParams(
            threshold=float(threshold),
            slope=float(slope),
            model=model,
            date=models_date_of_release[model]
        )
        results[model] = params.model_dump()
    return results

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Loading processed data from {args.input}")
    df = pd.read_csv(args.input)
    logging.info(f"Loaded {len(df)} records")
    
    logging.info("Fitting logistic curves by model")
    curve_params = fit_curves_by_model(df)
    logging.info(f"Fitted curves for {len(curve_params)} models")
    
    logging.info(f"Saving curve parameters to {args.output}")
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    # drop extension from filename if it exists
    if "." in filename:
        filename = filename.split(".")[0]
    data_utils.save_data({filename: curve_params}, output_dir)
    logging.info("Complete")

if __name__ == "__main__":
    main()
