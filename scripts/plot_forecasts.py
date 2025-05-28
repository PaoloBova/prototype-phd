"""
Generate plots from forecast data.
"""
import argparse
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import prototype_phd.data_utils as data_utils
from typing import Dict, List, Optional, Tuple, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate forecast plots")
    parser.add_argument("--inputs", required=True, help="Path pattern for input CSV files")
    parser.add_argument("--out", required=True, help="Output directory for plots")
    return parser.parse_args()

def read_forecast_data(input_pattern: str) -> Dict[str, pd.DataFrame]:
    """
    Read all forecast data matching the input pattern.
    
    Args:
        input_pattern: Glob pattern for input files
    
    Returns:
        Dictionary mapping file type to DataFrame
    """
    data = {}
    
    for file_path in glob(input_pattern):
        file_name = os.path.basename(file_path)
        file_type = file_name.split(".")[0]  # Remove extension
        
        data[file_type] = pd.read_csv(file_path)
        # Convert date columns to datetime
        if 'date' in data[file_type].columns:
            data[file_type]['date'] = pd.to_datetime(data[file_type]['date'])
        
        logging.info(f"Loaded {file_type} data with {len(data[file_type])} records")
    
    return data

def plot_characteristic_curves(ability_df: pd.DataFrame, output_dir: str):
    """
    Plot test-characteristic curves.
    
    Args:
        ability_df: DataFrame with ability forecasts
        output_dir: Directory to save plots
    """
    # Create a grid of difficulties for plotting
    x = np.linspace(-5, 20, 100)
    
    # Group by trend type and frequency
    for scenario in ability_df["scenario"].unique():
        scenario_df = ability_df[ability_df["scenario"] == scenario]
        
        # Get frequency and trend type from scenario
        trend_type, frequency = scenario.rsplit('_', 1)
        
        # Create figure for all years together
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate years from date for sorting and coloring
        scenario_df["year"] = scenario_df["date"].dt.year
        years = sorted(scenario_df["year"].unique())
        
        # Plot a curve for each year with color gradient
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(years), max(years))
        
        for _, row in scenario_df.iterrows():
            year = row["year"]
            threshold = row["threshold"]
            slope = row["slope"]
            
            # Calculate logistic curve
            y = 1.0 / (1.0 + np.exp(-slope * (x - threshold)))
            
            # Plot with color corresponding to year
            ax.plot(x, y, color=cmap(norm(year)), alpha=0.7)
        
        # Add colorbar for years
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm)
        cbar.set_label("Year")
        
        # Add labels and title
        ax.set_xlabel("Task Difficulty")
        ax.set_ylabel("Success Probability")
        ax.set_title(f"Forecasted Test-Characteristic Curves\nScenario: {trend_type}, Frequency: {frequency}")
        ax.grid(alpha=0.3)
        
        # Save the figure
        fig_name = f"characteristic_curves_{scenario}.png"
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Create facet plot by year
        years_per_plot = 2  # Show 2 years per facet
        num_plots = (len(years) + years_per_plot - 1) // years_per_plot
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        for i in range(num_plots):
            start_idx = i * years_per_plot
            end_idx = min((i + 1) * years_per_plot, len(years))
            years_subset = years[start_idx:end_idx]
            
            for year in years_subset:
                year_data = scenario_df[scenario_df["year"] == year]
                for _, row in year_data.iterrows():
                    threshold = row["threshold"]
                    slope = row["slope"]
                    
                    # Calculate logistic curve
                    y = 1.0 / (1.0 + np.exp(-slope * (x - threshold)))
                    
                    # Plot with label
                    month = row["date"].month
                    day = row["date"].day
                    date_label = f"{year}-{month:02d}-{day:02d}"
                    axes[i].plot(x, y, label=date_label)
            
            axes[i].set_title(f"Years: {years_subset[0]}-{years_subset[-1]}")
            axes[i].set_xlabel("Task Difficulty")
            axes[i].set_ylabel("Success Probability")
            axes[i].grid(alpha=0.3)
            axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        fig_name = f"characteristic_curves_{scenario}_by_year.png"
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

def plot_cost_trend(cost_df: pd.DataFrame, output_dir: str):
    """
    Plot cost trends.
    
    Args:
        cost_df: DataFrame with cost trends
        output_dir: Directory to save plots
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot doubling rate for each model
    sns.barplot(x="model", y="doubling_rate", data=cost_df, ax=ax)
    
    # Add labels and title
    ax.set_xlabel("Model")
    ax.set_ylabel("Cost Doubling Rate (difficulty units)")
    ax.set_title("Cost Doubling Rate by Model")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    # Save the figure
    fig_path = os.path.join(output_dir, "cost_doubling_rates.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create figure for R-squared values
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot R-squared for each model
    sns.barplot(x="model", y="r_squared", data=cost_df, ax=ax)
    
    # Add labels and title
    ax.set_xlabel("Model")
    ax.set_ylabel("R-squared")
    ax.set_title("Cost Model Fit Quality by Model")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    
    # Save the figure
    fig_path = os.path.join(output_dir, "cost_fit_quality.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_demand_windows(demand_df: pd.DataFrame, output_dir: str):
    """
    Plot demand windows as histograms.
    
    Args:
        demand_df: DataFrame with demand scenarios
        output_dir: Directory to save plots
    """
    # Convert date to datetime if not already
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    demand_df["year"] = demand_df["date"].dt.year
    
    # Extract unique combinations of ability scenario and budget scenario
    scenarios = demand_df.groupby(["ability_scenario", "budget_scenario"]).size().reset_index()[
        ["ability_scenario", "budget_scenario"]]
    
    for _, row in scenarios.iterrows():
        ability_scenario = row["ability_scenario"]
        budget_scenario = row["budget_scenario"]
        
        # Filter data for this scenario combination
        scenario_data = demand_df[
            (demand_df["ability_scenario"] == ability_scenario) & 
            (demand_df["budget_scenario"] == budget_scenario)
        ]
        
        # Create plot by year
        years = sorted(scenario_data["year"].unique())
        n_years = len(years)
        n_cols = min(3, n_years)
        n_rows = (n_years + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, year in enumerate(years):
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axes[row_idx, col_idx]
            
            year_data = scenario_data[scenario_data["year"] == year]
            
            # Create histogram of task allocations
            difficulties = []
            counts = []
            for _, group_row in year_data.iterrows():
                difficulties.append(group_row["difficulty"])
                counts.append(group_row["task_count"])
            
            # Plot histogram
            ax.bar(difficulties, counts)
            ax.set_title(f"Year: {year}")
            ax.set_xlabel("Task Difficulty")
            ax.set_ylabel("Task Count")
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_years, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            axes[row_idx, col_idx].set_visible(False)
        
        plt.suptitle(f"Task Demand Distribution\nAbility: {ability_scenario}\nBudget: {budget_scenario}")
        plt.tight_layout()
        
        # Save figure
        fig_name = f"demand_{ability_scenario}_{budget_scenario}.png"
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

def plot_bias_detection_metrics(sensitivity_df: pd.DataFrame, output_dir: str):
    """
    Plot bias and detection lag over time.
    
    Args:
        sensitivity_df: DataFrame with sensitivity results
        output_dir: Directory to save plots
    """
    # Convert date to datetime if not already
    sensitivity_df["date"] = pd.to_datetime(sensitivity_df["date"])
    sensitivity_df["year"] = sensitivity_df["date"].dt.year
    
    # Plot bias over time for each ability scenario and estimator
    for estimator in sensitivity_df["estimator"].unique():
        estimator_data = sensitivity_df[sensitivity_df["estimator"] == estimator]
        
        # Group by ability_scenario
        for ability_scenario in estimator_data["ability_scenario"].unique():
            scenario_data = estimator_data[estimator_data["ability_scenario"] == ability_scenario]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get all budget scenarios for consistent coloring
            budget_scenarios = sorted(scenario_data["budget_scenario"].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(budget_scenarios)))
            
            # Plot each budget scenario
            for budget_scenario, color in zip(budget_scenarios, colors):
                budget_data = scenario_data[scenario_data["budget_scenario"] == budget_scenario]
                
                # Group by year for averaging
                yearly_data = budget_data.groupby("year")["bias"].agg(["mean", "std"]).reset_index()
                
                # Plot line with error bars
                ax.errorbar(
                    yearly_data["year"], 
                    yearly_data["mean"], 
                    yerr=yearly_data["std"],
                    label=budget_scenario,
                    color=color,
                    capsize=5,
                    marker='o'
                )
                
            # Add reference line at zero bias
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            # Add labels and title
            ax.set_xlabel("Year")
            ax.set_ylabel(f"Bias ({estimator})")
            ax.set_title(f"Estimation Bias Over Time\nAbility: {ability_scenario}, Estimator: {estimator}")
            ax.legend(title="Budget Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(alpha=0.3)
            
            # Save figure
            fig_name = f"bias_{ability_scenario}_{estimator}.png"
            fig_path = os.path.join(output_dir, fig_name)
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            
    # Plot detection lag over time
    # Similar structure as bias plots
    for estimator in sensitivity_df["estimator"].unique():
        estimator_data = sensitivity_df[sensitivity_df["estimator"] == estimator]
        
        # Group by ability_scenario
        for ability_scenario in estimator_data["ability_scenario"].unique():
            scenario_data = estimator_data[estimator_data["ability_scenario"] == ability_scenario]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get all budget scenarios for consistent coloring
            budget_scenarios = sorted(scenario_data["budget_scenario"].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(budget_scenarios)))
            
            # Plot each budget scenario
            for budget_scenario, color in zip(budget_scenarios, colors):
                budget_data = scenario_data[scenario_data["budget_scenario"] == budget_scenario]
                
                # Group by year for averaging
                yearly_data = budget_data.groupby("year")["detection_lag"].agg(["mean", "std"]).reset_index()
                
                # Plot line with error bars
                ax.errorbar(
                    yearly_data["year"], 
                    yearly_data["mean"], 
                    yerr=yearly_data["std"],
                    label=budget_scenario,
                    color=color,
                    capsize=5,
                    marker='o'
                )
                
            # Add reference line at zero lag
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            
            # Add labels and title
            ax.set_xlabel("Year")
            ax.set_ylabel(f"Detection Lag (years)")
            ax.set_title(f"Detection Lag Over Time\nAbility: {ability_scenario}, Estimator: {estimator}")
            ax.legend(title="Budget Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(alpha=0.3)
            
            # Save figure
            fig_name = f"detection_lag_{ability_scenario}_{estimator}.png"
            fig_path = os.path.join(output_dir, fig_name)
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()

def create_combined_summary(sensitivity_df: pd.DataFrame, output_dir: str):
    """
    Create combined summary plot of bias and detection lag.
    
    Args:
        sensitivity_df: DataFrame with sensitivity results
        output_dir: Directory to save plots
    """
    # Convert date to datetime if not already
    sensitivity_df["date"] = pd.to_datetime(sensitivity_df["date"])
    sensitivity_df["year"] = sensitivity_df["date"].dt.year
    
    # Get unique combinations of ability_scenario and estimator
    scenario_estimator_pairs = sensitivity_df.groupby(["ability_scenario", "estimator"]).size().reset_index()[
        ["ability_scenario", "estimator"]]
    
    # Create a single combined plot for each scenario-estimator pair
    for _, row in scenario_estimator_pairs.iterrows():
        ability_scenario = row["ability_scenario"]
        estimator = row["estimator"]
        
        scenario_data = sensitivity_df[
            (sensitivity_df["ability_scenario"] == ability_scenario) & 
            (sensitivity_df["estimator"] == estimator)
        ]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get all budget scenarios for consistent coloring
        budget_scenarios = sorted(scenario_data["budget_scenario"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(budget_scenarios)))
        
        # Plot bias on first subplot
        for budget_scenario, color in zip(budget_scenarios, colors):
            budget_data = scenario_data[scenario_data["budget_scenario"] == budget_scenario]
            
            # Group by year for averaging
            yearly_data = budget_data.groupby("year")["bias"].agg(["mean", "std"]).reset_index()
            
            # Plot line with error bars
            ax1.errorbar(
                yearly_data["year"], 
                yearly_data["mean"], 
                yerr=yearly_data["std"],
                label=budget_scenario,
                color=color,
                capsize=5,
                marker='o'
            )
            
        # Add reference line at zero bias
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"Bias ({estimator})")
        ax1.set_title("Estimation Bias")
        ax1.grid(alpha=0.3)
        
        # Plot detection lag on second subplot
        for budget_scenario, color in zip(budget_scenarios, colors):
            budget_data = scenario_data[scenario_data["budget_scenario"] == budget_scenario]
            
            # Group by year for averaging
            yearly_data = budget_data.groupby("year")["detection_lag"].agg(["mean", "std"]).reset_index()
            
            # Plot line with error bars
            ax2.errorbar(
                yearly_data["year"], 
                yearly_data["mean"], 
                yerr=yearly_data["std"],
                label=budget_scenario,
                color=color,
                capsize=5,
                marker='o'
            )
            
        # Add reference line at zero lag
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Detection Lag (years)")
        ax2.set_title("Detection Lag")
        ax2.grid(alpha=0.3)
        
        # Add legend to the right of the second subplot
        ax2.legend(title="Budget Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add overall title
        fig.suptitle(f"Summary Metrics for {ability_scenario}, Estimator: {estimator}", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        fig_name = f"combined_summary_{ability_scenario}_{estimator}.png"
        fig_path = os.path.join(output_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

def main():
    """Main entry point."""
    data_utils.configure_logging_console()
    args = parse_args()
    
    logging.info(f"Reading forecast data from {args.inputs}")
    data = read_forecast_data(args.inputs)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    
    # Generate plots based on available data
    if any('ability' in file for file in data.keys()):
        ability_df = next(data[key] for key in data if 'ability' in key)
        logging.info("Plotting test-characteristic curves")
        plot_characteristic_curves(ability_df, args.out)
    
    if any('cost' in file for file in data.keys()):
        cost_df = next(data[key] for key in data if 'cost' in key)
        logging.info("Plotting cost trends")
        plot_cost_trend(cost_df, args.out)
    
    if any('demand' in file for file in data.keys()):
        demand_df = next(data[key] for key in data if 'demand' in key)
        logging.info("Plotting demand windows")
        plot_demand_windows(demand_df, args.out)
    
    if any('sensitivity' in file for file in data.keys()):
        sensitivity_df = next(data[key] for key in data if 'sensitivity' in key)
        logging.info("Plotting bias and detection metrics")
        plot_bias_detection_metrics(sensitivity_df, args.out)
        
        logging.info("Creating combined summary plots")
        create_combined_summary(sensitivity_df, args.out)
    
    logging.info(f"All plots saved to {args.out}")

if __name__ == "__main__":
    main()
