import logging
import numpy as np
import pandas as pd
import prototype_phd.data_utils as data_utils
import prototype_phd.methods.bootstrap as bootstrap
import prototype_phd.methods.mcdev as mcdev
import prototype_phd.stats as stats
import prototype_phd.utils as utils


setup_args = {"log_path": "logs/detection_rates_case_study.log",
              "data_dir_root": "data/detection_rates_case_study",
              "plots_dir_root": "plots/detection_rates_case_study",}
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

# Compute K, p_base, and B_max

# We split the times into powers of 2 for binning. We ignore the first 5 bins
# from our choice model since they only correspond to tasks from the SWAA task
# source, which isn't a good fit for our choice model.
# The longest task is in bin 17. So, we set K = 17 - 5 = 12.
# 12 bins isn't a lot, we we later consider a more finegrained choice model
# with K = 24 where each bin is in powers of sqrt(2).
# Since we start from the sixth bin, we set p_base as the average price (or
# generation cost) for a given model to complete tasks in the sixth bin.
# Finally, to compute B_max, we take the total generation cost of all tasks
# for that model (it doesn't matter if we exclude SWAA since the expenditure
# on these tasks is next to 0).

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
    max_val_po2 = int(np.ceil(np.log2(max_val)))
    bins = (2**np.array(range(max_val_po2+1))).astype(int)
    print("Bins:", bins)
    print("max_val:", max_val)
    # Optionally, label the bins with human-friendly labels.
    # For example, using the bin edges directly (or you can provide custom labels)
    df["log_bin"] = pd.cut(df["human_seconds"], bins=bins, include_lowest=True)
    # Get midpoints of the log_bin values
    df["log_bin_mid"] = df["log_bin"].apply(lambda x: np.mean([x.left, x.right]))
    # Get the nearest power of 2 (rounding down) for each log_bin
    df["log_bin_po2"] = df["log_bin"].apply(lambda x: int(np.floor(np.log2(x.left))))
    return df

df_case_study = process_data(df)

used_bins = np.sort(df_case_study["log_bin_po2"].unique())
# K should be per model but we can use all data for now
K = len(used_bins)
# TODO: p_base should be per model, but we can use the average for now
p_base = df_case_study[df_case_study["log_bin_po2"] == used_bins[0]]["generation_cost"].mean()
# TODO: B_max should be per model, but we can use the average for now
B_max = df_case_study["generation_cost"].sum() / len(df_case_study["model"].unique())

logging.info(f"Used bins: {used_bins}")
logging.info(f"B_max:  {B_max}, p_base: {p_base}")

scenarios = build_scenarios(K=K, p_base=p_base, B_max=B_max)
df_weights = mcdev.compute_allocations(scenarios)
# Create logistic regression wrapper
x_cols = ["log_human_seconds"]
y_col = "score_binarized"
logreg_config = stats.LogRegConfig(
    engine="scikit-learn",
    solver="lbfgs",
    C=1,
    max_iter=1000,
)
stats_fn = lambda idxs, df: bootstrap.analysis_logistic_regression(idxs,
                                                                    df,
                                                                    logreg_config,
                                                                    x_cols=x_cols,
                                                                    y_col=y_col)
# Run the bootstrap analysis
bootstrap_results = []
group_vars = ["model"]
gdfs = df_case_study.groupby(group_vars)
group_vars_weights = ["Budget"]
gdfs_weights = df_weights.groupby(group_vars_weights)

import tqdm as tqdm
for group, gdf in tqdm.tqdm(gdfs):
    for group_weights, gdf_weights in tqdm.tqdm(gdfs_weights):
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
            # sample_size=int(total_demand),
            sample_size=100,
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
                df_temp[col] = group_weights[i]
        for i, col in enumerate(group_vars):
            if len(group_vars) > 1:
                df_temp[col] = group[i]
            else:
                df_temp[col] = group[i]
        bootstrap_results.append(df_temp)
        
        # TODO: Remove break
        # break
    break

df_bootstrap = pd.concat(bootstrap_results)

data_to_save = {"df_bootstrap": df_bootstrap}
data_utils.save_data(data_to_save, data_dir=data_dir)

# Visualize the bootstrap distributions for each model and budget value
# plots = {}
# plot_group_vars = ["model", "Budget"]
# import matplotlib.pyplot as plt
# subplots, axs = plt.subplots(len(df_bootstrap[plot_group_vars]), 1, figsize=(10, 6))
# for group, gdf in df_bootstrap.groupby(plot_group_vars):
#     for col in gdf.columns:
#         if col in plot_group_vars:
#             continue
#         # Create a histogram of the bootstrap results
#         plt.figure(figsize=(10, 6))
#         plt.hist(gdf[col], bins=30, alpha=0.7, color='blue', edgecolor='black')
#         plt.title(f"Bootstrap Distribution for {group}")
#         plt.xlabel("Success Rate")
#         plt.ylabel("Frequency")
#         plt.grid()
#         plot_key = f"{group[0]}_{group[1]}_{col}"
#         plots[plot_key] = plt

# data_utils.save_plots(plots, plots_dir=f"{plots_dir}/case_study_plots")

# TODO: Make sure to run an analysis which estimates each bins success rate
# so that we can compute test senstitivity rates that way too
# TODO: Consider test sensitivities which look at the basic unit of task runs
# and thinks about how grouping them leads to thinking about how any particular
# task run might have a chance of misrepresenting what you think in general
# about task success rates in that bin.
# TODO: Consider using hierarchical bootstrap sampling by task_family and task_id
# At the moment, I'm sampling by task_run_id (for each alias)
# TODO: Consider whether the choice model using the budget estimated from the
# data may be ill-suited for recovering the original allocation. If so, then
# it's plausible that the allocations for tighter budget constraints doesn't
# exactly capture what might have been chosen. Also consider whether budget
# constraints should be in log space rather than percentile.
# TODO: Consider calibrating total demand for high budget against the total
# number of task runs for that model. This would help us to be more precise.
# Note: We appear to be off by a factor of 10. Increasing the sample size
# this much should help reduce spread of estimates but will take significantly
# longer to run.
# TODO: Refactor double loop into a single loop and compute the weights
# each time using model-specific data.
# TODO: Double check that log_human_seconds is computed properly