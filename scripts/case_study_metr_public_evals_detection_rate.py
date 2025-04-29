import logging
import numpy as np
import pandas as pd
import prototype_phd.data_utils as data_utils
import prototype_phd.methods.bootstrap as bootstrap
import prototype_phd.methods.mcdev as mcdev
import prototype_phd.stats as stats
import prototype_phd.utils as utils
import scipy as scipy
import tqdm as tqdm

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
    n = 10
    B_values = B_max * np.linspace(0 + 1/n, 1, n)
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

def compute_allocations_helper(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the allocations for the given DataFrame.
    
    Notes:
        - The DataFrame should contain the following columns:
            - log_bin_po2: The log2 bin for the task run times.
            - generation_cost: The generation cost for the task run.
            - model: The model name.
    """
    # TODO: How much does the number of used bins matter? What if a middle bin
    # goes unused? Does it make more sense to only exclude trailing empty bins?
    used_bins = np.sort(df["log_bin_po2"].unique())
    K = len(used_bins)
    p_base = df[df["log_bin_po2"] == used_bins[0]]["generation_cost"].mean()
    # If p_base is nan return 0
    if np.isnan(p_base):
        p_base = 1
    # Note: For some models, the generation cost is effectively 0. This implies
    # that budget constraints will never be binding. We could drop such models
    # from the analysis, but so that we can keep them in for now, we set
    # a minimum value for p_base and B_max.
    # TODO: Consider whether to use the largest B_max among models for all
    # models. Be aware that this could amplify bias in selection of longer tasks
    # for models where less budget was spent.
    # Note: Prices will always have to be model-specific.
    p_base = max(p_base, 1e-3)
    # Assume models can't have less than 1e-2 cost
    B_max = max(df["generation_cost"].sum(), 1e-1)
    if df['model'].unique()[0] == "gpt2":
        logging.info(f"costs: {df['generation_cost']}")
    logging.info(f"Model: {df['model'].unique()[0]}")
    logging.info(f"Used bins: {used_bins}")
    logging.info(f"K: {K}, p_base: {p_base}, B_max:  {B_max}")
    scenarios = build_scenarios(K=K, p_base=p_base, B_max=B_max)
    df_weights = mcdev.compute_allocations(scenarios)
    bin_item_mapping = {i: j for i, j in zip(range(1, K+1), used_bins)}
    # By construction we can always invert the item mapping.
    df_weights["Item_bin"] = df_weights["Item"].apply(lambda x: bin_item_mapping[x])
    return df_weights

def assign_mcdev_weights(df:pd.DataFrame,
                         df_weights:pd.DataFrame,
                         item_bin_col:str="log_bin_po2") -> np.ndarray:
    """Assign weights to items in `df` based on demand for item in `df_weights`.
    
    Notes:
    - The `df` DataFrame should contain the following columns:
        - item_index: The item index for row.
    - The `df_weights` DataFrame should contain the following columns:
        - Item: The item index.
        - Item_bin: The bin for the item .
        - Demand: The demand for the item.
    - The Item column in `df_weights` should not have any duplicates.
    - The `item_bin_col` parameter should be the name of the column in `df`
    that is used to identify the item of that row.
    - Returns None if the total demand or weights would effectively be 0.
    """
    total_demand = df_weights["Demand"].sum()
    if total_demand == 0:
        # We don't need data when Budget is or near zero
        return None
    assert total_demand > 0
    # The item column should not have any duplicates.
    assert df_weights["Item"].is_unique
    allocations = df_weights["Demand"] / total_demand
    item_bins = df_weights["Item_bin"]
    allocations_by_item_bin = dict(zip(item_bins.values, allocations.values))
    weights = df[item_bin_col].map(lambda x: allocations_by_item_bin.get(x, 0)).values
    weights_sum = weights.sum()
    if weights_sum == 0:
        # We can ignore this group if no tasks are relevant
        return None
    assert weights_sum > 0
    weights = weights / weights_sum
    return weights

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
    logging.info(f"Bins: {bins}")
    logging.info(f"max_val: {max_val}")
    # Optionally, label the bins with human-friendly labels.
    # For example, using the bin edges directly (or you can provide custom labels)
    df["log_bin"] = pd.cut(df["human_seconds"], bins=bins, include_lowest=True)
    # Get lower and upper bounds of the bins
    df["log_bin_left"] = df["log_bin"].apply(lambda x: x.left)
    df["log_bin_right"] = df["log_bin"].apply(lambda x: x.right)
    # Get midpoints of the log_bin values
    df["log_bin_mid"] = df["log_bin"].apply(lambda x: np.mean([x.left, x.right]))
    # Get the nearest power of 2 (rounding down) for each log_bin
    df["log_bin_po2"] = df["log_bin"].apply(lambda x: int(np.floor(np.log2(x.left))))
    return df

df_case_study = process_data(df)

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
case_vars = ["Budget"]
bootstrap_config_default = {"n_bootstrap": 1000,
                           "analysis_funcs": [stats_fn],
                           "sample_size": 100,
                           "random_state": 1,
                           }

for group, gdf in tqdm.tqdm(gdfs):
    df_weights = compute_allocations_helper(gdf)
    gdfs_weights = df_weights.groupby(case_vars)
    for case, gdf_weights in tqdm.tqdm(gdfs_weights):
        weights = assign_mcdev_weights(gdf, gdf_weights)
        if weights is None:
            # If weights is None, we can ignore this group
            continue
        bootstrap_config = bootstrap.BootstrapConfig(
            **bootstrap_config_default,
            # sample_size=int(gdf_weights["Demand"].sum()),
            weights=weights)
        args = bootstrap.BootstrapDataInput(df=gdf, bootstrap_config=bootstrap_config)
        df_temp = bootstrap.run_bootstrap(args)
        # Add group variables to the results
        for i, col in enumerate(case_vars):
            df_temp[col] = case[i]
        for i, col in enumerate(group_vars):
            df_temp[col] = group[i]
        bootstrap_results.append(df_temp)

df_bootstrap = pd.concat(bootstrap_results)
data_to_save = {"df_bootstrap": df_bootstrap}
data_utils.save_data(data_to_save, data_dir=data_dir)

# Alternative approach
# Use the success rate of task runs per bin in the original dataframe.
# Then for each MCDEV scenario, we can compute the test sensitivity rates
# for each bin analytically.
# The test sensitivity rates are the probability of choosing at least a threshold
# number of successful task runs in that bin given the budget allocations and
# given that the model has a success rate above that threshold.
x_pct = 0.5

analytical_results = []
success_rates_results = []
group_vars = ["model"]
gdfs = df_case_study.groupby(group_vars)
case_vars = ["Budget"]
for group, gdf in tqdm.tqdm(gdfs):
    df_weights = compute_allocations_helper(gdf)
    gdfs_weights = df_weights.groupby(case_vars)
    df_success_rates = gdf.groupby("log_bin_po2", observed=True)["score_binarized"].mean()
    success_rates_results.append(df_success_rates.reset_index())
    for case, gdf_weights in tqdm.tqdm(gdfs_weights):
        # Skip if budget is 0
        if case[0] == 0:
            continue
        # If total demand is 0, we can ignore this group (happens when Budget is close to 0)
        if gdf_weights["Demand"].sum() == 0:
            continue
        demands = gdf_weights["Demand"].values
        num_draws = demands
        cutoffs = np.ceil(num_draws * x_pct).astype(int)
        success_rates =  gdf_weights["Item_bin"].apply(lambda x: df_success_rates[x])
        item_bin = gdf_weights["Item_bin"].values
        item1 = gdf_weights["Item_bin"].values[0]
        gdf_weights["sensitivity_rate"] = [scipy.stats.binom.sf(c - 1, n, p)
                                            for c,n,p in zip(cutoffs, num_draws, success_rates)]
        gdf_weights["success_rate"] = success_rates
        gdf_weights["num_draws"] = num_draws
        gdf_weights["cutoff"] = cutoffs
        gdf_weights["success_threshold"] = x_pct
        # Add group variables to the results
        for i, col in enumerate(case_vars):
            gdf_weights[col] = case[i]
        for i, col in enumerate(group_vars):
            gdf_weights[col] = group[i]
        analytical_results.append(gdf_weights)

df_analytical = pd.concat(analytical_results)
df_success_rates = pd.concat(success_rates_results)
data_to_save = {"df_analytical": df_analytical,
                "df_success_rates": df_success_rates}
data_utils.save_data(data_to_save, data_dir=data_dir)

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
# TODO: Double check that log_human_seconds is computed properly