import logging
import numpy as np
import pandas as pd
import prototype_phd.data_utils as data_utils
import prototype_phd.methods.bootstrap as bootstrap
import prototype_phd.methods.mcdev as mcdev
import prototype_phd.utils as utils


setup_args = {"log_path": "logs/detection_rates.log"}
sim_id, commit, data_dir, plots_dir = data_utils.setup_project(**setup_args)

logging.info(f"Simulation ID: {sim_id}")
logging.info(f"Commit: {commit}")

# Load external data
external_data_dir = "external_data/detection_rates"
file_path = f"{external_data_dir}/metr_public_evals_all_runs.jsonl"
data = data_utils.read_ndjson(file_path)
df = pd.DataFrame(data)
logging.info(f"External data columns: {df.columns}")


# Methodology
# Here, we analyse the case study data in different plausible counterfactual
# settings with a tighter budget constraint.

# To do this, we sample rows with replacement (a bootstrap sample). Each row
# has a different weight.

# We compute the weights given a choice model over task families
# Specifically, we create a mapping between task times in log2 space and the
# number of task runs demanded. To ensure we sample a number of observations
# equal to these demands on average, we weight each task run in the dataset
# by the demand for tasks of that length normalized by the total demand. We
# also set the number of draws within each booststrap sample to be equal to the
# total demand.


def build_scenarios(p_base=1.0, B_max=1000, K=15):
    # We need prices to increase exponentially with k.
    scenario_config = mcdev.ScenarioConfig(
            scenario_name="Exponential Increase",
            scenario_func=mcdev.scenario_exponential,
            K=K)
    k_vec = np.arange(1, K + 1)
    p = scenario_config.scenario_func(k_vec, p_base, 1)
    gamma = 5 / (1 + np.exp(-0.5 * (np.arange(K) - K/2)))
    psi = mcdev.scenario_exponential(np.arange(K), 1, 0.5)
    B_values = B_max * np.linspace(0, 1, 10)
    alpha = 0.0
    variable_parameters = {
        "B": B_values.tolist(),
        "p": [p],
        "psi": [psi],
        "gamma": [gamma],
        "alpha": [alpha],
        "scenario_config": [scenario_config],
    }
    configs = [mcdev.DemandConfig(**d)
               for d in utils.dict_list(variable_parameters)]
    return configs

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # Filter out SWAA task source
    df = df.copy()
    df = df[df["task_source"] != "SWAA"]
    df["human_seconds"] = df["human_minutes"] * 60
    df["log_human_seconds"] = np.log2(df["human_seconds"])
    # Bin the human_seconds into log2 bins.
    max_val = df["human_seconds"].max()
    n_bins = np.log2(max_val).round().astype(int)
    bins = np.logspace(0, np.log2(max_val), n_bins, True, 2)
    # Optionally, label the bins with human-friendly labels.
    # For example, using the bin edges directly (or you can provide custom labels)
    df["log_bin"] = pd.cut(df["human_seconds"], bins=bins, include_lowest=True)
    # Get midpoints of the log_bin values
    df["log_bin_mid"] = df["log_bin"].apply(lambda x: np.mean([x.left, x.right]))
    
    # TODO: Improve processing so we match original data analysis
    
    return df

df_case_study = process_data(df)

scenarios = build_scenarios()
df_weights = mcdev.compute_allocations(scenarios)
# Create logistic regression wrapper
x_cols = ["log_human_seconds"]
y_col = "score_binarized"
stats_fn = lambda a,b: bootstrap.analysis_logistic_regression(a,b, x_cols=x_cols, y_col=y_col)
# Modify your bootstrap analysis with these safeguards
def stats_fn_with_safeguards(a, b, x_cols=x_cols, y_col=y_col):
    # Check for minimum counts of each outcome class
    if (b[y_col].sum() < 3) or (len(b) - b[y_col].sum() < 3):
        # Not enough samples in one class - return null result or flag
        return {"converged": False, "warning": "insufficient_variation"}
    
    # Check for perfect separation before running regression
    # (Simple check: see if any value of x perfectly predicts y)
    for col in x_cols:
        for val in b[col].unique():
            y_subset = b[b[col] == val][y_col]
            if (len(y_subset) > 0) and ((y_subset.mean() == 0.0) or (y_subset.mean() == 1.0)):
                # Perfect separation detected
                return {"converged": False, "warning": "perfect_separation"}
    
    # If we passed the checks, proceed with the regression
    return bootstrap.analysis_logistic_regression(a, b, x_cols=x_cols, y_col=y_col)
# Run the bootstrap analysis
bootstrap_results = []
group_vars = ["model"]
gdfs = df_case_study.groupby(group_vars)
group_vars_weights = ["Budget"]
gdfs_weights = df_weights.groupby(group_vars_weights)
for group, gdf in gdfs:
    for group_weights, gdf_weights in gdfs_weights:
        total_demand = gdf_weights["Demand"].sum()
        if total_demand == 0:
            # We don't need data when Budget is or near zero
            continue
        assert total_demand > 0
        allocations = gdf_weights["Demand"] / total_demand
        item_index = gdf_weights["Item"]
        # Extract the values for allocations and item_index which are pandas series groupby objects
        allocations_by_item_index = dict(zip(item_index.values, allocations.values))
        gdf["item_index"] = gdf["log_human_seconds"].astype(int)
        weights = gdf["item_index"].map(lambda x: allocations_by_item_index.get(x, 0)).values
        weights_sum = weights.sum()
        if weights_sum == 0:
            # We can ignore this group if no tasks are relevant
            continue
        assert weights_sum > 0
        weights = weights / weights_sum
        bootstrap_config = bootstrap.BootstrapConfig(
            n_bootstrap=10000,
            analysis_funcs=[stats_fn],
            sample_size=int(total_demand),
            weights=weights,
            random_state = 1,
        )
        # Run the bootstrap analysis
        bootstrap_input = bootstrap.BootstrapDataInput(
            df=gdf,
            bootstrap_config=bootstrap_config,
        )
        df_temp = bootstrap.run_bootstrap(bootstrap_input)
        # Add group variables to the results
        for i, col in enumerate(group_vars_weights):
            if len(group_vars_weights) > 1:
                df_temp[col] = group_weights[i]
            else:
                df_temp[col] = group_weights
        for i, col in enumerate(group_vars):
            if len(group_vars) > 1:
                df_temp[col] = group[i]
            else:
                df_temp[col] = group
        bootstrap_results.append(df_temp)
        
        break

df_bootstrap = pd.concat(bootstrap_results)

# TODO: Investigate edge cases with getting bootstrap samples for the case
# study data, e.g. when all sampled task runs are successful/unsuccessful.
# Suggestion, make sure parameters are set so that at full budget we sample
# as mant task runs as in original dataset for each model. When budget is low
# this may happen organically, in which case we will want to skip logistic
# regression and set the estimator to be the max value tested for (if all
# successful) or the min value (if all unsuccessful).
# TODO: Create plots for the bootstrap results