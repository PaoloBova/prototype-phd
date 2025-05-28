"""
Data preprocessing for forecast detection case study.
"""
import argparse
import logging
import numpy as np
import pandas as pd
import prototype_phd.data_utils as data_utils
from typing import Any, Dict
from .schemas import RawDataRecord, ProcessedDataRecord

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess detection data")
    parser.add_argument("--config", required=True, help="Path to config file")
    return parser.parse_args()

def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess raw data according to configuration.
    
    Args:
        df: Raw data frame
        config: Configuration dictionary
    
    Returns:
        Processed DataFrame
    """

    # Filter data if specified in config
    df = data_utils.filter_data(df, config)
    df = df.copy()

    # Create derived columns
    df["human_seconds"] = df["human_minutes"] * 60
    df["log2_human_seconds"] = np.log2(df["human_seconds"])
    
    # Create difficulty bins based on human seconds
    df = data_utils.bin_data_by_power(df, "human_seconds", base=2)
    
    # Also load in release_dates.yaml
    release_dates = data_utils.read_config(config["release_dates_path"])
    df_dates = pd.DataFrame(release_dates["date"].items(), columns=["alias", "date"])
    df = pd.merge(df, df_dates, on="alias", how="left")
    logging.info(f"Number of rows missing a date: {df['date'].isna().sum()}")

    return df

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    config = data_utils.read_config(args.config)
    
    logging.info(f"Loading data from {config['data_path']}")
    data = data_utils.read_ndjson(config["data_path"])
    try:
        [RawDataRecord(**record).model_dump() for record in data]
    except Exception as e:
        logging.warning(f"Data validation failed (likely due to incomplete data): {e}")

    df = pd.DataFrame(data)
    logging.info(f"Loaded {len(df)} records")
    
    logging.info("Preprocessing data")
    df_processed = preprocess_data(df, config)
    logging.info(f"Processed data has {len(df_processed)} records")
    try:
        df_processed.apply(lambda row: ProcessedDataRecord(**row).model_dump(), axis=1)
    except Exception as e:
        logging.warning(f"Data validation failed (likely due to incomplete data): {e}")

    
    logging.info(f"Saving processed data to {config['output_dir']}")
    data_utils.save_data({"processed_data": df_processed}, config["output_dir"])
    logging.info("Preprocessing complete")

if __name__ == "__main__":
    main()
