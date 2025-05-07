"""
Case study for detection rates in METR public evaluations.

This script is used to analyze the detection rates of task runs in the METR
public evaluations dataset. It uses the MCDEV model to compute demand
allocations based on the generation cost of tasks and the success rates
of task runs. The script also includes functions for bootstrapping
the analysis and for plotting the results.

Methodology
----------
Here, we analyse the case study data in different plausible counterfactual
settings with a tighter budget constraint.

To do this, we sample rows with replacement (a bootstrap sample). Each row
has a different weight.

We compute the weights given a choice model over task families
Specifically, we create a mapping between task times in log2 space and the
number of task runs demanded. To ensure we sample a number of observations
equal to these demands on average, we weight each task run in the dataset
by the demand for tasks of that length normalized by the total demand. We
also set the number of draws within each booststrap sample to be equal to the
total demand.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prototype_phd.data_utils as data_utils
import prototype_phd.methods.bootstrap as bootstrap
import prototype_phd.methods.mcdev as mcdev
import prototype_phd.stats as stats
import prototype_phd.utils as utils
import scipy as scipy
import seaborn as sns
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

@utils.multi
def compute_demands_by_budget(df: pd.DataFrame,
                              mode: str = "simple",
                              bin_col: str = "bin_power",
                              cost_col: str = "generation_cost") -> pd.DataFrame:
    """
    Multimethod for deriving demands by budget.

    Parameters
    ----------
    df : pd.DataFrame
        The input data, which may contain multiple observations per “item.”
    mode : str, optional
        Dispatch key. Possible values: "simple", "mcdev".
    bin_col : str, optional
        Name of the column for the item’s bin.
    cost_col : str, optional
        Name of the column for each observation’s cost.

    Returns
    -------
    pd.DataFrame
        The structure depends on the selected method, but typically includes
        columns like:
          - Item : integer index (1-based) for each bin.
          - Item_bin : the bin label.
          - Budget : the budget tested.
          - Demand : the count (or demand) of observations included.
    """
    return mode

@utils.method(compute_demands_by_budget, "simple")
def compute_demands_by_budget(df: pd.DataFrame,
                              mode: str = "simple",
                              bin_col: str = "bin_power",
                              cost_col: str = "generation_cost") -> pd.DataFrame:
    """
    Approach
    --------
    
    Uses a simple, bin-based approach for deriving demand counts under a range
    of budgets. Asummes that a shrinking budget hits the largest bins first.
    1. Sort items by bin and cost, compute a cumulative cost, and ensure bin
    column is categorical.
    2. For each budget, filter items whose cumulative cost is ≤ budget.
    3. Group by bins (including those not present in the subset) to count how
    many times each item is bought.
    4. Collect these counts in a final DataFrame for analysis of aggregate
    demand per bin at varying budgets."""
    # Exclude trailing empty bins; keep all bins between the first and last bin.
    bin_sup = df[bin_col].max(skipna=True)
    bin_inf = df[bin_col].min(skipna=True)
    # Assume bins are integers
    bins_consecutive = np.array(range(bin_inf, bin_sup + 1)).astype(int)
    K = len(bins_consecutive)
    full_budget = df[cost_col].sum()
    n = 10
    B_values = full_budget * np.linspace(0 + 1/n, 1, n)

    df_ordered = df.sort_values([bin_col, cost_col]).copy()
    df_ordered["cumulative_cost"] = df_ordered[cost_col].cumsum()
    df_ordered[bin_col] = pd.Categorical(df_ordered[bin_col], categories=bins_consecutive)

    records = []
    for budget in B_values:
        df_constrained = df_ordered[df_ordered["cumulative_cost"] <= budget]
        demands = df_constrained.groupby(bin_col, observed=False)[cost_col].count()
        col_names = ["Item", "Item_bin", "Budget", "Demand"]
        obs = zip(np.arange(1, K + 1), bins_consecutive, [budget] * K, demands.values)
        records.extend({c: v for c, v in zip(col_names, row)} for row in obs)

    return pd.DataFrame(records)

@utils.method(compute_demands_by_budget, "mcdev_hardcoded")
def compute_demands_by_budget(df: pd.DataFrame,
                              mode: str = "mcdev_hardcoded",
                              bin_col: str = "bin_power",
                              cost_col: str = "generation_cost") -> pd.DataFrame:
    """
    MCDEV-based approach for deriving demands by budget. 
    This uses scenario configs, exponential scaling, etc.
    """

    # TODO: How much does the number of used bins matter? What if a middle bin
    # goes unused? Does it make more sense to only exclude trailing empty bins?
    # Assume we always have a contiguous set of used bins.
    used_bins = np.sort(df[bin_col].unique())
    K = len(used_bins)

    
    # In this approach, we don't care about the exact data.
    # We compute prices as follows:
    prices = [2**k for k in range(K)]
    p_base = 1
    # Assume demands are the same for all bins.
    avg_demand = 40
    avg_demand2 = np.mean(df.groupby("bin_power")["generation_cost"].count())
    B_max = avg_demand * np.sum(prices)

    # Build MCDEV scenarios, then compute allocations:
    scenario_config = mcdev.ScenarioConfig(
        scenario_name="Exponential Increase",
        scenario_func=mcdev.scenario_exponential,
        K=K
    )
    k_vec = np.arange(1, K + 1)
    p = scenario_config.scenario_func(k_vec, p_base, 1)
    # Currently, I need to hardcode the gamma values to get a shape which
    # matches the data more closely. Later, we'll do a maximum likelihood
    # estimation to get the gamma values (and the psi values) given the prices
    # and demands from the data.
    # I find that multipling a logistic function by a linear function gives
    # a good approximation of the data under the choice model.
    gamma = 10 * 5 / (1 + np.exp(-0.5 * (np.arange(K) - K/2)))
    gamma = np.linspace(0.25, 4.75, K) * gamma
    # gamma = np.ones(K)
    psi = mcdev.scenario_exponential(np.arange(K), 1, 0.5)
    # psi = 0.5 * np.log(np.arange(1, K + 1))
    # prices = np.arange(1, K + 1)
    # B_max = avg_demand * np.sum(prices)
    p = np.array(prices)
    n = 10
    # Create a range of points exponentially spaced between 0 and B_max
    B_values = np.logspace(0, np.log2(B_max), num=n, endpoint=True, base=2)    
    # B_values = B_max * np.linspace(0 + 1/n, 1, n)
    alpha = 0
    variable_parameters = {
        "B": B_values.tolist(),
        "p": [p],
        "psi": [psi],
        "gamma": [gamma],
        "alpha": [alpha],
        "scenario_config": [scenario_config],
    }
    scenarios = [mcdev.DemandConfig(**d)
                 for d in utils.dict_list(variable_parameters)]
    
    df_weights = mcdev.compute_allocations(scenarios)
    bin_item_mapping = {i: j for i, j in zip(range(1, K+1), used_bins)}
    # By construction we can always invert the item mapping.
    df_weights["Item_bin"] = df_weights["Item"].map(bin_item_mapping)

    return df_weights

@utils.method(compute_demands_by_budget, "mcdev_calibrated_v1")
def compute_demands_by_budget(df: pd.DataFrame,
                              mode: str = "mcdev_calibrated_v1",
                              bin_col: str = "bin_power",
                              cost_col: str = "generation_cost") -> pd.DataFrame:
    """
    MCDEV-based approach for deriving demands by budget.
    This uses scenario configs, exponential scaling, etc.
    We calibrate the scenario parameters based on some of the data in df
    """
    
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
    
    # TODO: How much does the number of used bins matter? What if a middle bin
    # goes unused? Does it make more sense to only exclude trailing empty bins?
    # Assume we always have a contiguous set of used bins.
    used_bins = np.sort(df["bin_power"].unique())
    K = len(used_bins)
    p_base = df[df["bin_power"] == used_bins[0]]["generation_cost"].mean()
    prices = [df[df["bin_power"] == used_bin]["generation_cost"].mean()
              for used_bin in used_bins]
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
    # Note: We want to calibrate our budget and prices such that at full budget
    # each model samples ~100 observations for each bin.
    # We can achieve this by setting a max_budget for each model equal to
    # the top price of that model times B_max (most expensive model) / p_top (most expensive model).
    # This would ensure that every model can buy B_max / p_top of the most expensive
    # task runs when at full budget.
    # We want B_max / p_top to be around 50.
    p_top = prices[-1]
    p_max = np.max(prices)

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
    df_weights = mcdev.compute_allocations(configs)
    bin_item_mapping = {i: j for i, j in zip(range(1, K+1), used_bins)}
    # By construction we can always invert the item mapping.
    df_weights["Item_bin"] = df_weights["Item"].apply(lambda x: bin_item_mapping[x])
    return df_weights

@utils.method(compute_demands_by_budget, "mcdev_calibrated_v2")
def compute_demands_by_budget(df: pd.DataFrame,
                              mode: str = "mcdev_calibrated_v2",
                              bin_col: str = "bin_power",
                              cost_col: str = "generation_cost") -> pd.DataFrame:
    """
    MCDEV-based approach for deriving demands by budget.
    This uses scenario configs, exponential scaling, etc.
    We calibrate the scenario parameters based on some of the data in df
    """
    
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
    
    # Exclude trailing empty bins; keep all bins between the first and last bin.
    bin_sup = df["bin_power"].max(skipna=True)
    bin_inf = df["bin_power"].min(skipna=True)
    # Assume bins are integers
    bins_consecutive = np.array(range(bin_inf, bin_sup + 1)).astype(int)
    K = len(bins_consecutive)

    # To compute prices: Use average costs for each bin
    prices = [df[df["bin_power"] == bin]["generation_cost"].mean()
              for bin in bins_consecutive]
    prices = np.array(prices)
    # Note: Unfortunately, the actual costs lead to optimal choices where all
    # expenditure goes to the longest tasks (no matter the budget). This fits
    # the data very poorly. The current hypothesis is that this is because
    # the team anticipates that the costs will follow a predictable trend (as
    # they cannot know the costs before choosing which tasks to run).
    # Caveat: Might be able to tell based on costs for previous models.

    B_max = df["generation_cost"].sum()
    n_constraints = 10
    B_values = B_max * np.linspace(0 + 1/n_constraints, 1, n_constraints)

    # Choice of gamma and psi is handcrafted for now. Future approaches will
    # attempt to estimate them from data.
    gamma = 10 * 5 / (1 + np.exp(-0.5 * (np.arange(K) - K/2)))
    gamma = np.linspace(0.25, 4.75, K) * gamma
    psi = mcdev.scenario_exponential(np.arange(K), 1, 0.5)
    # We set alpha = 0 for simplicity.
    alpha = 0.0
    # Specify legacy scenario config that we don't use anymore.
    scenario_config = mcdev.ScenarioConfig(
            scenario_name="Exponential Increase",
            scenario_func=mcdev.scenario_exponential,
            K=K)

    variable_parameters = {
        "B": B_values.tolist(),
        "p": [prices],
        "psi": [psi],
        "gamma": [gamma],
        "alpha": [alpha],
        "scenario_config": [scenario_config],
    }
    configs = [mcdev.DemandConfig(**d)
               for d in utils.dict_list(variable_parameters)]
    df_weights = mcdev.compute_allocations(configs)
    bin_item_mapping = {i: j for i, j in zip(range(1, K+1), bins_consecutive)}
    # By construction we can always invert the item mapping.
    df_weights["Item_bin"] = df_weights["Item"].apply(lambda x: bin_item_mapping[x])
    return df_weights

@utils.method(compute_demands_by_budget, "mcdev_calibrated_v3")
def compute_demands_by_budget(df: pd.DataFrame,
                              mode: str = "mcdev_calibrated_v3",
                              bin_col: str = "bin_power",
                              cost_col: str = "generation_cost") -> pd.DataFrame:
    """
    MCDEV-based approach for deriving demands by budget.
    This uses scenario configs, exponential scaling, etc.
    We calibrate the scenario parameters based on some of the data in df
    """
    
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
    
    # Exclude trailing empty bins; keep all bins between the first and last bin.
    bin_sup = df[bin_col].max(skipna=True)
    bin_inf = df[bin_col].min(skipna=True)
    # Assume bins are integers
    bins_consecutive = np.array(range(bin_inf, bin_sup + 1)).astype(int)
    K = len(bins_consecutive)

    # To compute prices: Fit an OLS regression to the cost data. Use
    # predicted costs for each bin. We have previously found that an exponential
    # fit has better goodness of fit than a linear fit. So, run OLS regression
    # on log2(cost) vs bin. Then, use the predicted values to compute prices.
    gdf = df.groupby(bin_col)[cost_col].mean().reset_index()
    gdf["log_cost"] = np.log2(gdf[cost_col])
    # Fit OLS regression
    x = gdf[bin_col].values
    y = gdf["log_cost"].values
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    # Fit a line if there are enough points
    if len(x) > 1:
        slope, intercept = np.polyfit(x, y, deg=1)
    else:
        # If not enough points, use the average cost for the first bin
        slope = 0
        intercept = np.log2(df[df[bin_col] == bins_consecutive[0]][cost_col].mean())
    # Compute the predicted costs
    prices = 2**(intercept + slope * bins_consecutive)
    # Set psi to be correlated with the predicted costs (handcrafted)
    psi = 2 **((intercept + slope * bins_consecutive) / 2)

    B_max = df[cost_col].sum()
    n_constraints = 10
    B_values = B_max * np.linspace(0 + 1/n_constraints, 1, n_constraints)

    # Choice of gamma is handcrafted for now. Future approaches will
    # attempt to estimate them from data.
    gamma = np.ones(K)
    gamma = 10 * 5 / (1 + np.exp(-0.5 * (np.arange(K) - K/4)))
    gamma = np.linspace(0.25, 4.75, K) * gamma
    # We set alpha = 0 for simplicity.
    alpha = 0.0
    # Specify legacy scenario config that we don't use anymore.
    scenario_config = mcdev.ScenarioConfig(
            scenario_name="Exponential Increase",
            scenario_func=mcdev.scenario_exponential,
            K=K)

    variable_parameters = {
        "B": B_values.tolist(),
        "p": [prices],
        "psi": [psi],
        "gamma": [gamma],
        "alpha": [alpha],
        "scenario_config": [scenario_config],
    }
    configs = [mcdev.DemandConfig(**d)
               for d in utils.dict_list(variable_parameters)]
    df_weights = mcdev.compute_allocations(configs)
    bin_item_mapping = {i: j for i, j in zip(range(1, K+1), bins_consecutive)}
    # By construction we can always invert the item mapping.
    df_weights["Item_bin"] = df_weights["Item"].apply(lambda x: bin_item_mapping[x])
    
    # Sanity check that expenditures sum to B_values
    y = df_weights.groupby("Budget")["Expenditure"].sum().reset_index()["Expenditure"].values
    x = B_values
    logging.info(f"Expenditures: {y}")
    logging.info(f"Budgets: {x}")
    # Check that the expenditures are close to the budgets
    assert np.allclose(x, y, rtol=0.1, atol=0.1)
    return df_weights

def assign_mcdev_weights(df:pd.DataFrame,
                         df_weights:pd.DataFrame,
                         item_bin_col:str="bin_power") -> np.ndarray:
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
    # Filter out human and gpt2 models (cost data is not comparable to others)
    df = df[df["model"] != "human"]
    df = df[df["model"] != "gpt2"]
    df["human_seconds"] = df["human_minutes"] * 60
    df["log_human_seconds"] = np.log2(df["human_seconds"])
    df = data_utils.bin_data_by_power(df, "human_seconds", base=2**0.5)
    return df

def run_boostrap_helper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run bootstrap analysis.
    """

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
    gdfs = df.groupby(group_vars)
    case_vars = ["Budget"]
    bootstrap_config_default = {"n_bootstrap": 1000,
                            "analysis_funcs": [stats_fn],
                            "sample_size": 100,
                            "random_state": 1,
                            }
    choice_method = "mcdev_calibrated"

    for group, gdf in tqdm.tqdm(gdfs):
        df_weights = compute_demands_by_budget(gdf, choice_method)
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

    return df_temp

def run_analytic_helper(df: pd.DataFrame,
                        x_pct:float=0.5,
                        choice_method:str="mcdev_calibrated_v3") -> pd.DataFrame:
    """
    Run the analytical helper.

    Use the success rate of task runs per bin in the original dataframe.
    Then for each MCDEV scenario, we can compute the test sensitivity rates
    for each bin analytically.
    The test sensitivity rates are the probability of choosing at least a threshold
    number of successful task runs in that bin given the budget allocations and
    given that the model has a success rate above that threshold.
    """

    analytical_results = []
    success_rates_results = []
    group_vars = ["model"]
    gdfs = df.groupby(group_vars)
    case_vars = ["Budget"]
    bin_col = "bin_power"
    for group, gdf in tqdm.tqdm(gdfs):
        df_weights = compute_demands_by_budget(gdf, mode=choice_method)
        gdfs_weights = df_weights.groupby(case_vars)
        # Turn bin_power into a categorical variable
        # Exclude trailing empty bins; keep all bins between the first and last bin.
        bin_sup = df[bin_col].max(skipna=True)
        bin_inf = df[bin_col].min(skipna=True)
        # Assume bins are integers
        bins_consecutive = np.array(range(bin_inf, bin_sup + 1)).astype(int)
        gdf[bin_col] = pd.Categorical(gdf[bin_col], categories=bins_consecutive)
        df_success_rates = gdf.groupby(bin_col, observed=False)["score_binarized"].mean()
        success_rates_results.append(df_success_rates.reset_index())
        for case, gdf_weights in tqdm.tqdm(gdfs_weights):
            # Skip if budget is 0
            if case[0] == 0:
                continue
            # If total demand is 0, we can ignore this group (happens when Budget is close to 0)
            if gdf_weights["Demand"].sum() == 0:
                continue
            demands = gdf_weights["Demand"].values
            num_draws = np.floor(demands).astype(int)
            cutoffs = np.ceil(num_draws * x_pct).astype(int)
            success_rates =  gdf_weights["Item_bin"].apply(lambda x: df_success_rates[x])
            logging.info(f"cutoffs: {cutoffs}")
            logging.info(f"num_draws: {num_draws}")
            logging.info(f"success_rates: {success_rates}")
            gdf_weights["sensitivity_rate"] = [scipy.stats.binom.sf(c - 1, n, p) if n >= 1 else 0
                                                for c,n,p in zip(cutoffs, num_draws, success_rates)]
            logging.info(f"sensitivity_rates: {gdf_weights['sensitivity_rate']}")
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

def plot_avg_prices_by_model_subplots(df: pd.DataFrame,
                                      bin_col: str = "bin_power",
                                      cost_col: str = "generation_cost",
                                      model_col: str = "model"):
    """
    Create one subplot per model to show (bin, average price) scatter points
    and a fitted line if enough points remain after removing NaNs.
    Up to four subplots per row are used; extra subplots remain blank if
    there aren't enough models to fill them.
    """
    grouped = df.groupby([model_col, bin_col])[cost_col].mean().reset_index()
    models = grouped[model_col].unique()
    n_models = len(models)

    import math
    ncols = 4
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 4*nrows),
                             sharex=False, sharey=False,
                             constrained_layout=True)

    # If only a single subplot is created, wrap it in a list for consistency
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        subdf = grouped[grouped[model_col] == model]

        x_vals = subdf[bin_col].astype(float).values
        y_vals = subdf[cost_col].values
        mask = ~np.isnan(x_vals) & ~np.isnan(y_vals) & np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        ax.scatter(x_vals, y_vals, alpha=0.6, label=f"{model}")

        # Fit a line if enough valid points remain
        if len(x_vals) > 1 and x_vals.min() != x_vals.max():
            slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_fit = slope * x_range + intercept
            ax.plot(x_range, y_fit, linestyle='-', label=f"{model} fit")

            # Compute R^2 manually
            y_pred = slope * x_vals + intercept
            y_mean = y_vals.mean()
            ss_total = np.sum((y_vals - y_mean) ** 2)
            ss_res = np.sum((y_vals - y_pred) ** 2)
            r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

            # Annotate with R^2
            ax.text(0.04, 0.9,
                    f"R² = {r2:.3f}",
                    transform=ax.transAxes,
                    ha="left", va="center",
                    fontsize=9)

        # Add extra padding for clarity
        ax.set_title(f"Model: {model}", pad=15)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Average Price")
        ax.legend()

    # Hide any remaining axes (if fewer models than subplots)
    for ax in axes[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig

def plot_method_comparison_allocations(config: mcdev.PlotConfig,
                                       method_col="Method") -> object:
    """
    Compare demand allocations for different methodologies in a single plot.

    We assume config.df contains columns:
      - Item (int) : item index
      - Demand (float) : computed allocations/demands
      - Budget (float) : the budget for that scenario
      - <method_col> (str) : the methodology/scenario name (e.g. "simple", "mcdev")

    Plots a grouped bar (Item on x-axis, Demand on y-axis, hue by method),
    with columns faceted by Budget. 
    """
    df = config.df.copy()
    y_var = config.y_var

    # Create a grouped bar chart, faceting by budget.
    # Each facet shows a comparison of demands across “Method”.
    g = sns.catplot(
        data=df,
        x="Item",
        y=y_var,
        hue=method_col,
        col="Budget",
        kind="bar",
        height=3,
        aspect=1.2,
        col_wrap=4,
        sharey=False,
        palette="coolwarm"
    )
    g.set_axis_labels("Item", y_var)
    g.set_titles("Budget = {col_name}")
    plot_title = f"{y_var} Comparison by Method & Budget: {config.plot_group}"
    g.figure.suptitle(plot_title, y=1.04)

    # Tighten the layout
    plt.tight_layout()
    return g.figure

def run_debug_plots(df: pd.DataFrame) -> dict:
    """
    Run debug plots to help calibrate the case study.
    """
    # Plotting code to help debug the code and calibrate the choice model
    group_vars = ["model"]
    gdfs = df.groupby(group_vars)
    figs = {}
    for group, gdf in tqdm.tqdm(gdfs):
        df_weights = compute_demands_by_budget(gdf, mode="mcdev_calibrated_v3")
        
        df_simple = compute_demands_by_budget(gdf, mode="simple")
        df_simple["Method"] = "simple"
        df_mcdev = compute_demands_by_budget(gdf, mode="mcdev_calibrated_v3")
        df_mcdev["Method"] = "mcdev_calibrated"
        df_combined = pd.concat([df_simple, df_mcdev], ignore_index=True)
        plot_group =  dict(zip(group_vars, group))
        y_var = "Demand"
        pc = mcdev.PlotConfig(
                plot_group=plot_group,
                df=df_combined,
                y_var=y_var,
            )
        fig1 = plot_method_comparison_allocations(pc)
        gdf["log_prices"] = np.log2(gdf["generation_cost"])
        fig2 = plot_avg_prices_by_model_subplots(gdf, cost_col="log_prices")
        figs[f"group_{group}_method_comparison"] = fig1
        figs[f"group_{group}_avg_prices"] = fig2

    # Plot average prices for each model and task bin
    df["log_prices"] = np.log2(df["generation_cost"])
    fig = plot_avg_prices_by_model_subplots(df, cost_col="log_prices")
    figs["log_price_trends"] = fig
    data_utils.save_plots(figs, plots_dir=plots_dir)

df_case_study = process_data(df)

run_debug_plots(df_case_study)

run_analytic_helper(df_case_study)


# THE LONG LIST OF TODOS
# ------------------------------------------------------

# TODO: Consider estimating mcdev model parameters from data

# TODO: Assume that counterfactual allocations will be based on a choice model
# with parameters given by an assumed scaling law for the inference cost of tasks
# of different length and the time+money cost of creating the tasks.
# TODO: Another way of doing this is to assume that you take away enough budget
# to eliminate using tasks from the last bin. Keep in mind that under a MCDEV
# choice model, this wouldn't actually be low enough to usually eliminate the
# last bin as fewer earlier tasks would be used instead.
# TODO: Assume naively that allocations are predetermined under some scaling law
# model without using any generation cost data for the tasks. Which scaling law
# model? How about our simple exp2 model where the cost of a task is 2^k? We
# also calculate the main budget as the sum of the anticipated costs of the
# demanded task runs in each bin. This will not match the actual generation cost
# data at all, but it will be a good starting point for the analysis. When I
# do want to use the actual data, I should start with anticipated costs based
# on a linear fit to the generation cost and human cost data. Again, the assumed
# budget will be the sum of the anticipated costs rather than the actual costs.
# If computing number of draws, calculate the allocation-weighted price across
# bins and divide the assumed budget by this price. Or just use the total
# demand implied by the allocations directly (that would be easier).
# TODO: What does the need to abstract away from the actual data mean for
# the theory? It suggests that the budget constraints are not as easy to predict
# as might be needed for applying the theory to forecasting. Specifically, how
# bad a gap in evaluation budgets might be is unclear. It might be that the
# most valuable tests are not the most expensive ones.
# TODO: We need to create an adversarial model of company behaviour when there
# are conditional risk thresholds in place. What type of testing will they
# choose to do prior to release (how does this respond to safety buffers)?
# How could they report more favourable results than they should? What could
# they do if they don't have direct control over the evaluations but there is
# asymmetric information?
# TODO: Start with a naive cutting down of task runs starting with longest tasks
# first (or most expensive). Then, suggest choice model which reduces the impact
# on the longest task runs at the cost of percision for the earlier runs.

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