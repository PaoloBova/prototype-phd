import prototype_phd.data_utils as data_utils
import prototype_phd.stats as stats

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

def custom_x_axis(ax, skip_ticks=1):
    # Define tick positions (in seconds) and labels.
    # 0 sec, then 1, 2, 4, 8, 15, 30 sec,
    # then 1, 2, 4, 8, 15, 30 min (converted to sec),
    # then 1, 2, 4, 8 hr (converted to sec).
    ticks = [0, 1, 2, 4, 8, 15, 30,
            60, 120, 240, 480, 900, 1800,
            3600, 7200, 14400, 28800, 54000, 108000]
    tick_labels = ['0', '1', '2', '4', '8', '15', '30',
                '1m', '2m', '4m', '8m', '15m', '30m',
                '1h', '2h', '4h', '8h', '15h', '30h']
    # If you want a symlog scale (so near zero it’s linear),
    # which is useful because log(0) is undefined, uncomment:
    # If skip_ticks=1,we skip every second tick
    ticks = ticks[::skip_ticks]
    tick_labels = tick_labels[::skip_ticks]
    ax.set_xscale('symlog')

    # Use a scale which counts in seconds in powers of 2 (so a custom log scale)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(0, ticks[-1])
    return None

def compute_clustered_se(subset):
    """Compute clustered standard error for the given group."""
    # Subset the data for the current task_source and model

    if len(subset) > 1:  # Ensure there are enough data points
        overall_mean = subset["score_binarized"].mean()
        overall_var = np.var(subset["score_binarized"])
        overall_se = np.sqrt(overall_var / len(subset))
        
        # Compute the triple summation
        clusters = subset.groupby("task_family")["score_binarized"]
        cluster_variance = 0
        for cluster, group in clusters:
            deviations = group - overall_mean
            deviations = deviations.values
            cluster_variance += (deviations[:, None] @ deviations[None, :]).sum()
        
        # Compute the clustered standard error
        clustered_se = np.sqrt(
            overall_se ** 2 +
            1 / (len(subset) ** 2) * cluster_variance
        ) 
    else:
        # If there's only one data point, return a standard error of 0
        clustered_se = 0
    return clustered_se

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

# Plot the average successes of the different models on each task source.

# Group data by task_source and model
task_sources = df["task_source"].unique()
models = df["model"].unique()
task_sources.sort()
models.sort()
gdfs = df.groupby(["task_source", "model"])
results = gdfs["score_binarized"].mean().unstack()

# Group data by task_source and model
task_sources = df["task_source"].unique()
models = df["model"].unique()
task_sources.sort()
models.sort()
gdfs = df.groupby(["task_source", "model"])
results = gdfs["score_binarized"].mean().unstack()

# Initialize a dictionary to store clustered standard errors
clustered_se = {group: compute_clustered_se(subset) for group, subset in gdfs}


# Plot the average successes with error bars
plt.figure(figsize=(10, 6))
for model in models:
    y = results[model]
    yerr = [clustered_se.get((task_source, model), 0) for task_source in task_sources]  # Get SEs or default to 0
    plt.errorbar(task_sources, y, yerr=yerr, fmt='o', label=model)

# Add labels and legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Task source")
plt.ylabel("Average success rate")
plt.title("Average Success Rates with Clustered Standard Errors")
plt.show()

# Plot the average success of the different models for only the hcast task source
# Filter the data for the specific task source
source_x = "HCAST"  # Replace with the actual task source name
subset = df[df["task_source"] == source_x]

# Group by model and calculate means and standard errors
model_groups = subset.groupby("model")
means = model_groups["score_binarized"].mean()
unadjusted_se = model_groups["score_binarized"].std() / np.sqrt(model_groups.size())

# Sort groups by mean success rate
means = means.sort_values(ascending=True)
unadjusted_se = unadjusted_se[means.index]

# Compute clustered standard errors
clustered_se = {}
for model, group in model_groups:
    clustered_se[model] = compute_clustered_se(group)

# Prepare data for plotting
models = means.index
x_positions = np.arange(len(models))  # Categorical x-axis positions
unadjusted_se_values = unadjusted_se.values
clustered_se_values = [clustered_se[model] for model in models]

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))

# Plot unadjusted standard errors
ax.errorbar(
    x_positions - 0.1,  # Slightly shift to the left
    means,
    yerr=unadjusted_se_values,
    fmt="o",
    label="Unadjusted SE",
    color="blue",
)

# Plot clustered standard errors
ax.errorbar(
    x_positions + 0.1,  # Slightly shift to the right
    means,
    yerr=clustered_se_values,
    fmt="o",
    label="Clustered SE",
    color="red",
)

# Customize the plot
ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=45, ha="right")
ax.set_xlabel("Models")
ax.set_ylabel("Average Success Rate")
ax.set_title(f"Comparison of Standard Errors for Task Source: {source_x}")
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()


# Plot the average success rates of each model as we vary task time baselines.
fig, ax = plt.subplots(figsize=(10, 6))
human_minutes = df["human_minutes"].unique()
df["human_seconds"] = df["human_minutes"] * 60
human_seconds = df["human_seconds"].unique()
human_seconds.sort()
gdfs = df.groupby(["human_seconds", "task_source"])
results = gdfs["score_binarized"].mean().unstack()
task_sources = df["task_source"].unique()
task_sources.sort()
for task_source in task_sources:
    plt.scatter(human_seconds, results[task_source], label=task_source)
plt.legend()
plt.xlabel("Task time baseline")
plt.ylabel("Average success rate")
# Move legend outside grid
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
custom_x_axis(ax)

# Plot the average success rates of each model as we vary task time baselines.
# We need around 15 subplots, one for each model.
# We first need to bin the human_seconds into log2 bins.
# We then calculate the average success rate for each model in each bin.

max_val = df["human_seconds"].max()
n_bins = np.log2(max_val).round().astype(int)
bins = np.logspace(0, np.log2(max_val), n_bins, True, 2)
# Optionally, label the bins with human-friendly labels.
# For example, using the bin edges directly (or you can provide custom labels)
df["log_bin"] = pd.cut(df["human_seconds"], bins=bins, include_lowest=True)
# Get midpoints of the log_bin values
df["log_bin_mid"] = df["log_bin"].apply(lambda x: np.mean([x.left, x.right]))

# Now you can group by the new log_bin_mid column instead of human_seconds,
# then calculate the average success rate in each bin.
gdfs = df.groupby(["log_bin_mid", "alias"])
results = gdfs["score_binarized"].mean().unstack()

gdfs2 = df.groupby(["human_seconds", "alias"])
results2 = gdfs2["score_binarized"].mean().unstack()
trials_grouped = gdfs2["score_binarized"].count().unstack()


# Plot for each alias using the bin midpoints as x-axis locations.
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
aliases = df["alias"].unique()
aliases.sort()
# Also, create a dataframe containing the average success rates for each alias
alias_df = df.groupby(["alias"])["score_binarized"].mean()
# We will use this to store thre threshold values for each alias.
thresholds = []
thresholds_se = []
for i, alias in enumerate(aliases):
    ax = axes[i // 3, i % 3]
    # Since the grouping was done by log_bin, extract the average success rates
    # and use bin_midpoints as x, assuming the order matches.
    bin_results = results[alias].dropna()
    # All bars should be the same width in log scale.
    # They should also be transparent and each alias should use a different color
    # to make it easier to distinguish them.
    ax.bar(results.index, bin_results, width=np.diff(bins), align="center",alpha=0.4, label=alias,color=plt.cm.viridis(i / len(aliases)))
    # # Add logistic regression line
    x = results2.index.astype(float)
    y = results2[alias]
    trials = trials_grouped[alias]
    # Make sure to drop all rows with at least one NA value.
    sdf = pd.DataFrame({"x": x, "y": y, "trials": trials}).dropna().reset_index(drop=True)
    x, y, trials = sdf["x"], sdf["y"], sdf["trials"]
    x_log2 = np.log2(x)

    # Fit model using robust_logistic_fit.
    result = stats.robust_logistic_fit(x_log2, y, freq_weights=trials)
    logging.info("Fitted model summary:\n%s", result.summary())

    # Run diagnostics.
    diagnostics = stats.run_diagnostics(result)

    # Compare link functions.
    metrics, best_link = stats.compare_link_functions(x_log2, y, freq_weights=trials)
    logging.info("Suggested best link: %s", best_link) 
    
    # Plot fitted logistic curve.
    x_plot = np.logspace(x_log2.min(), x_log2.max(), 200, True, 2)
    X_plot = sm.add_constant(pd.DataFrame(np.log2(x_plot), columns=['x']))
    fitted_probs = result.predict(X_plot)
    ax.plot(x_plot, fitted_probs, 'r-', label='Fitted logistic curve')
    
    # Add time horizon score annotation
    log_threshold = stats.compute_threshold(result)
    threshold = 2 ** log_threshold
    # Add vertical line for threshold and a shaded CI region.
    # Threshold should be written in seconds, minutes, or hours, whichever is most appropriate.
    if threshold < 60:
        time_horizon_label = f"{threshold:.2f} seconds"
    elif threshold < 3600:
        time_horizon_label = f"{threshold/60:.2f} minutes"
    else:
        time_horizon_label = f"{threshold/3600:.2f} hours"
    ax.axvline(threshold, color='k', linestyle='--', label=time_horizon_label)
    # Compute standard error of the threshold using the delta method.
    cov = result.cov_params()
    beta0, beta1 = result.params.iloc[0], result.params.iloc[1]
    var_log_threshold = (1/beta1**2) * cov.iloc[0, 0] + (beta0**2/(beta1**4)) * cov.iloc[1, 1] - \
                    (2 * beta0/(beta1**3)) * cov.iloc[0, 1]
    var_threshold = 2 ** (2 * log_threshold) * np.log(2)**2 * var_log_threshold
    se_threshold = np.sqrt(var_threshold)
    # Compute the 95% confidence interval for the threshold.
    ci_lower = threshold - 1.96 * se_threshold
    ci_upper = threshold + 1.96 * se_threshold
    ax.fill_betweenx([0, 1], ci_lower, ci_upper, color='gray', alpha=0.5)
    
    thresholds.append(threshold)
    thresholds_se.append(se_threshold)
 
    ax.set_xlabel("Task time baseline")
    ax.set_ylabel("Average success rate")
    ax.legend()
    custom_x_axis(ax, skip_ticks=2)
    # Set y axis between 0 and 1
    ax.set_ylim(0, 1 + 0.05)

alias_df["threshold"] = thresholds
alias_df["threshold_se"] = thresholds_se
# alias_df["threshold_ci_lower"] = alias_df["threshold"] - 1.96 * alias_df["threshold_se"]
# alias_df["threshold_ci_upper"] = alias_df["threshold"] + 1.96 * alias_df["threshold_se"]

# # Add alias release dates to the dataframe.
# # Fake them for now
# alias_release_dates = {}
# alias_df["release_date"] = alias_df["alias"].map(alias_release_dates)

# # Plot the thresholds for each alias with respect to the release date.
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, alias in enumerate(aliases):
#     release_date = alias_df["release_date"].iloc[i]
#     # Convert release date to a number (e.g. days since epoch)
#     # For now, use the index as a placeholder
#     x = i
#     y = alias_df["threshold"].iloc[i]
#     yerr = alias_df["threshold_se"].iloc[i]
#     ax.errorbar(x, y, yerr=yerr, fmt='o', label=alias, color=plt.cm.viridis(i / len(aliases)))
# ax.set_xticks(range(len(aliases)))
# ax.set_xticklabels(aliases, rotation=45, ha='right')
# ax.set_xlabel("Alias")
# ax.set_ylabel("Threshold (seconds)")
# ax.set_title("Thresholds for each alias with respect to release date")

plt.show()

# Counterfactual budget constraints

# We need to redo the above plots for different bootstrap samples.
# Each set of boostrap samples follows a different budget constraint.
# With no budget constraint, we have the full dataset, i.e. all rows have
# equal weight when sampled.
# With a budget constraint, we sample rows with replacement, but each row
# has a different weight. We have a few approaches to calculating weights.
# Approach 1: The weight depends on the inverse of the generation
# cost of the row.
# Approach 2: The weight depends on the inverse of the task time baseline (or
# the inverse of the human baseline cost which should give similar results).
# We also consider a completely different sampling approach, where given a 
# budget constraint, we sample the average number of rows, moving sequentially
# from shorter to longer task time baselines until we reach the budget.
# We then calculate the average success rate for each model for each bootstrap
# sample on each task family.
