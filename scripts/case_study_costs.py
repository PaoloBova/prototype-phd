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


human_minutes = df["human_minutes"].unique()
df["human_seconds"] = df["human_minutes"] * 60
human_seconds = df["human_seconds"].unique()
human_seconds.sort()
task_sources = df["task_source"].unique()
task_sources.sort()


# Plot the average costs and total costs with the human time baseline
# First without binning

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df.groupby(["human_seconds", "alias"]) 
results = gdfs["generation_cost"].count().unstack()
for alias in results.columns:
    plt.scatter(results.index, results[alias], label=alias)
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Demand")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='center left')
plt.tight_layout()
custom_x_axis(ax)


# fig, ax = plt.subplots(figsize=(10, 6))
# gdfs = df.groupby(["human_seconds", "task_source"])
# results = gdfs["generation_cost"].mean().unstack()
# for task_source in task_sources:
#     plt.scatter(human_seconds, results[task_source], label=task_source)
# plt.legend()
# plt.xlabel("Task time baseline")
# plt.ylabel("Average price")
# # Move legend outside grid
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# custom_x_axis(ax)

# fig, ax = plt.subplots(figsize=(10, 6))
# results = gdfs["generation_cost"].sum().unstack()
# for task_source in task_sources:
#     plt.scatter(human_seconds, results[task_source], label=task_source)
# plt.legend()
# plt.xlabel("Task time baseline")
# plt.ylabel("Exp")
# # Move legend outside grid
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# custom_x_axis(ax)

# Clearly we'll need to bin the data to get a better view of the costs
# Use the log2 scale for binning

max_val = df["human_seconds"].max()
n_bins = np.log2(max_val).round().astype(int)
bins = np.logspace(0, np.log2(max_val), n_bins, True, 2)
# Optionally, label the bins with human-friendly labels.
# For example, using the bin edges directly (or you can provide custom labels)
df["log_bin"] = pd.cut(df["human_seconds"], bins=bins, include_lowest=True)
# Get midpoints of the log_bin values
df["log_bin_mid"] = df["log_bin"].apply(lambda x: np.mean([x.left, x.right]))
# Create more readable bin labels for display
log_bin_intervals_sorted = sorted(df["log_bin"].unique(), key=lambda x: x.left)
readable_labels = [f"2^{int(np.log2(interval.left))}-2^{int(np.log2(interval.right))}"
                   for interval in log_bin_intervals_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df.groupby(["log_bin_mid"]) 
results = gdfs["generation_cost"].mean().reset_index(drop=False)
# Plot with categorical x-axis
# plt.scatter(readable_labels, results["generation_cost"])
plt.scatter(results['log_bin_mid'], results["generation_cost"])
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Average price")
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# Adjust layout to prevent label cutoff
plt.tight_layout()
custom_x_axis(ax)

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df.groupby(["log_bin_mid"]) 
results = gdfs["generation_cost"].sum().reset_index(drop=False)

plt.scatter(results['log_bin_mid'], results["generation_cost"])
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Expenditure")
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# Adjust layout to prevent label cutoff
plt.tight_layout()
custom_x_axis(ax)


fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df.groupby(["log_bin_mid"]) 
results = gdfs["generation_cost"].count().reset_index(drop=False)

plt.scatter(results['log_bin_mid'], results["generation_cost"])
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Demand")
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# Adjust layout to prevent label cutoff
plt.tight_layout()
custom_x_axis(ax)

# fig, ax = plt.subplots(figsize=(12, 6))
# gdfs = df.groupby(["human_seconds"]) 
# results = gdfs["generation_cost"].count().reset_index(drop=False)
# # Plot with categorical x-axis
# # plt.bar(readable_labels, results["generation_cost"])
# plt.scatter(results['human_seconds'], results["generation_cost"])
# plt.xlabel("Task time baseline (seconds, log2 scale)")
# plt.ylabel("Demand")
# plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# # Adjust layout to prevent label cutoff
# plt.tight_layout()
# custom_x_axis(ax)

# fig, ax = plt.subplots(figsize=(12, 6))
# gdfs = df.groupby(["human_seconds"]) 
# results = gdfs["generation_cost"].sum().reset_index(drop=False)
# # Plot with categorical x-axis
# # plt.bar(readable_labels, results["generation_cost"])
# plt.scatter(results['human_seconds'], results["generation_cost"])
# plt.xlabel("Task time baseline (seconds, log2 scale)")
# plt.ylabel("Expenditure")
# plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# # Adjust layout to prevent label cutoff
# plt.tight_layout()
# custom_x_axis(ax)


# Plot cumulative demands and expenditures with task time baseline
# Helpful to avoid issues with artifacts caused by binning


# fig, ax = plt.subplots(figsize=(12, 6))
# gdfs = df.groupby(["human_seconds"]) 
# results = gdfs["generation_cost"].sum().reset_index(drop=False)
# # Plot with categorical x-axis
# # plt.bar(readable_labels, results["generation_cost"])
# plt.scatter(results['human_seconds'], results["generation_cost"].cumsum())
# plt.xlabel("Task time baseline (seconds, log2 scale)")
# plt.ylabel("Cumulative expenditure")
# plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# # Adjust layout to prevent label cutoff
# plt.tight_layout()
# custom_x_axis(ax)


# fig, ax = plt.subplots(figsize=(12, 6))
# gdfs = df.groupby(["human_seconds"]) 
# results = gdfs["generation_cost"].count().reset_index(drop=False)
# # Plot with categorical x-axis
# # plt.bar(readable_labels, results["generation_cost"])
# plt.scatter(results['human_seconds'], results["generation_cost"].cumsum())
# plt.xlabel("Task time baseline (seconds, log2 scale)")
# plt.ylabel("Cumulative demand")
# plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# # Adjust layout to prevent label cutoff
# plt.tight_layout()
# custom_x_axis(ax)


# Plot cumulative demand and expenditure after filterout out SWAA task source

df_filtered = df[df["task_source"] != "SWAA"]
# outlier_mask = df_filtered["task_id"].isin(df_filtered["task_id"].value_counts().index[df_filtered["task_id"].value_counts() < 200])
# A human_seconds group is an outlier if it has more than 200 observations
outlier_mask = df_filtered.groupby("human_seconds")["task_id"].transform(lambda x: len(x) > 200)
df_filtered = df_filtered[~outlier_mask]

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds"]) 
results = gdfs["generation_cost"].sum().reset_index(drop=False)

plt.scatter(results['human_seconds'], results["generation_cost"].cumsum())
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Cumulative expenditure")
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# Adjust layout to prevent label cutoff
plt.tight_layout()
custom_x_axis(ax)

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds"]) 
results = gdfs["generation_cost"].count().reset_index(drop=False)

plt.scatter(results['human_seconds'], results["generation_cost"].cumsum())
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Cumulative demand")
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# Adjust layout to prevent label cutoff
plt.tight_layout()
custom_x_axis(ax)

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds"]) 
results = gdfs["generation_cost"].mean().reset_index(drop=False)
# Plot with categorical x-axis
# plt.scatter(readable_labels, results["generation_cost"])
plt.scatter(results['human_seconds'], results["generation_cost"])
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Average price")
plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
# Adjust layout to prevent label cutoff
plt.tight_layout()
custom_x_axis(ax)



# Plot cumulative demand and expenditure after filterout out SWAA task source
# grouped and colored by model or alias
df_filtered = df[df["task_source"] != "SWAA"]
# outlier_mask = df_filtered["task_id"].isin(df_filtered["task_id"].value_counts().index[df_filtered["task_id"].value_counts() < 200])
# A human_seconds group is an outlier if it has more than 200 observations
outlier_mask = df_filtered.groupby("human_seconds")["task_id"].transform(lambda x: len(x) > 200)
df_filtered = df_filtered[~outlier_mask]

fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds", "alias"]) 
results = gdfs["generation_cost"].sum().unstack()
for alias in results.columns:
    plt.scatter(results.index, results[alias].cumsum(), label=alias)
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Cumulative expenditure")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='center left')
plt.tight_layout()
custom_x_axis(ax)


fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds", "alias"]) 
results = gdfs["generation_cost"].count().unstack()
for alias in results.columns:
    plt.scatter(results.index, results[alias].cumsum(), label=alias)
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Cumulative demand")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='center left')
plt.tight_layout()
custom_x_axis(ax)


fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds", "alias"]) 
results = gdfs["generation_cost"].mean().unstack()
for alias in results.columns:
    plt.scatter(results.index, results[alias], label=alias)
    # Plot best fit regression line of y ~ log(x)
    x = np.log2(results.index)
    y = results[alias]
    # X = sm.add_constant(x)
    X = x
    sm_model = sm.OLS(y, X)
    sm_results = sm_model.fit()
    y_pred = sm_results.predict(X)
    plt.plot(results.index, y_pred)
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Average price")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='center left')
plt.tight_layout()
custom_x_axis(ax)


fig, ax = plt.subplots(figsize=(12, 6))
gdfs = df_filtered.groupby(["human_seconds", "alias"]) 
results = gdfs["generation_cost"].mean().unstack()
for alias in results.columns:
    plt.scatter(np.log2(results.index), np.log2(results[alias]), label=alias)
    # Plot best fit regression line of y ~x
    x = results.index
    y = results[alias]
    # X = sm.add_constant(x)
    X = x
    sm_model = sm.OLS(y, X)
    sm_results = sm_model.fit()
    y_pred = sm_results.predict(X)
    plt.plot(np.log2(results.index), np.log2(y_pred))
plt.xlabel("Task time baseline (seconds, log2 scale)")
plt.ylabel("Average price (log2 scale)")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='center left')
plt.tight_layout()
# custom_x_axis(ax)

# The results for similar models appear to be similar 

# These average prices appear to show that as the task time baseline increases,
# the average price increases, but so too does the variance in the prices. This
# might potentially be modelled as the average price increasing as well as a
# thicker tail of the distribution. This could be modelled as a log-normal
# distribution with a higher variance. The log-normal distribution is a
# distribution of a random variable whose logarithm is normally distributed.
# This is a common distribution for prices and is used in finance to model
# stock prices. The log-normal distribution is defined as:
# If X is log-normally distributed, then Y = ln(X) is normally distributed.
# The log-normal distribution is defined by two parameters: mu and sigma.
# The mean of the log-normal distribution is given by:
# mu + sigma^2 / 2
# The variance of the log-normal distribution is given by:
# (e^(sigma^2) - 1) * e^(2*mu + sigma^2)
# The log-normal distribution is defined as:
# f(x) = (1 / (x * sigma * sqrt(2 * pi))) * e^(-(ln(x) - mu)^2 / (2 * sigma^2))
# We can make mu and sigma functions of the human time baseline
# and then fit the data to the log-normal distribution using maximum likelihood
# estimation.

# IMPORTANT

# I should keep in mind that how the task source was created was likely by
# searching for a large set of tasks that appeared to vary in difficult and then
# getting humans to test how difficult they were. Ingoring any processing on
# the resulting set of tasks, they likely ran a similar number of runs per task.
# However, even if they intended for a uniform task difficulty distribution at
# the beginning, they ended up with a somewhat asymmetric bell-shaped distribution
# of task difficulties. This is likely due to the fact that the tasks are
# not uniformly distributed in difficulty and that the humans are not
# uniformly distributed in their ability to solve the tasks. This would suggest
# that rather than choosing to allocate demand across tasks of different difficulties,
# that instead they received a distribution of tasks and made use of them in
# a roughly uniform manner.

# Alternative approach:
# Assume they target a mean difficulty for an otherwise known distribution of
# task difficulty. They know their expected cost will equal budget. If budget
# were restricted they'd chose a lower mean difficulty.

# If we instead assume the original approach is active on task source creation
# too, then:
# Task source creators could fairly easily target teh difficulties of tasks they
# wanted to create in principle, but faced different search costs and had
# different marginal utilities and satiation preferences for tasks of different
# levels of difficulty. For example, coming up with more difficult tasks is
# more rewarding, harder to do, and you arguably need to cover a much larger
# complex space of very different tasks to avoid obvious criticism. If all of
# these things are true, then the way you distribution your search resources
# can be explained by an MCDEV model and directly shape the generation protocol
# for tasks of different difficulties.

# I'm noting that allowing for some noise at the higher end of the distribution
# the distribution of task difficulties is arguably similar to what we get from
# the MCDEV model where we have a sustained increase in frequency followed by
# a sharp drop off. This is not a perfect fit, but it is a reasonable
# approximation.

plt.close("all")

# Plot histogram of average human seconds per task id
fig, ax = plt.subplots(figsize=(12, 6))
df_filtered = df[df["task_source"] == "HCAST"]
gdfs = df_filtered.groupby(["task_id"])
results = gdfs["human_seconds"].mean().reset_index(drop=False)
results["human_seconds_log"]= np.log2(results["human_seconds"])
plt.hist(results["human_seconds_log"], bins=50)
plt.xlabel("Average task time baseline (seconds, log2 scale)")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='center left')
plt.tight_layout()

# Plot the same histograms for each alias using seaborn
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 6))
df_filtered = df[df["task_source"] == "HCAST"]
gdfs = df_filtered.groupby(["task_id", "alias"])
results = gdfs["human_seconds"].mean().reset_index(drop=False)
results["human_seconds_log"]= np.log2(results["human_seconds"])
sns.histplot(data=results, x="human_seconds_log", hue="alias", bins=50, kde=True,
             element="step", fill=False, cumulative=True)

fig, ax = plt.subplots(figsize=(12, 6))
df_filtered = df[df["task_source"] == "HCAST"]
df_filtered["human_seconds_log"]= np.log2(df_filtered["human_seconds"])
sns.histplot(data=df_filtered, x="human_seconds_log", hue="alias", bins=50, kde=True,
             element="step", fill=False, cumulative=True)

fig, ax = plt.subplots(figsize=(12, 6))
df_filtered = df[df["task_source"] == "HCAST"]
df_filtered["human_seconds_log"]= np.log2(df_filtered["human_seconds"])
sns.histplot(data=df_filtered, x="human_seconds_log", hue="alias", bins=50, kde=True,
             element="step", fill=False)

# # Plot count of observations for each task_id as orderd by human_seconds
# # for each alias do a different subplot
# fig, axs = plt.subplots(4, 4, figsize=(12, 6))
# df_filtered = df[df["task_source"] == "HCAST"]
# gdfs = df_filtered.groupby(["human_seconds", "alias"])
# results = gdfs["human_seconds"].count().unstack()
# # Do a different subplot per alias
# for i, alias in enumerate(results.columns):
#     ax = axs[i // 4, i % 4]
#     # color by alias
#     colors = {alias: plt.colormaps.get_cmap("tab10")(i)
#               for i, alias in enumerate(df_filtered.alias.unique())}
#     ax.scatter(results.index, results[alias], color=colors[alias])
#     ax.set_title(alias)
#     ax.set_ylabel("Demand")
#     ax.set_xlabel("Average task time baseline (seconds, log2 scale)")
#     ax.set_xticklabels(results.index, rotation=45, ha='right')
#     custom_x_axis(ax)
# plt.tight_layout()

def compare_hist_fits(df):


    # Plot histogram of average human seconds per task id
    fig, ax = plt.subplots(figsize=(12, 6))
    df_filtered = df[df["task_source"] == "HCAST"]
    gdfs = df_filtered.groupby(["task_id"])
    results = gdfs["human_seconds"].mean().reset_index(drop=False)
    results["human_seconds_log"] = np.log2(results["human_seconds"])

    # Get the histogram data
    hist, bin_edges = np.histogram(results["human_seconds_log"], bins=50, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Plot histogram
    plt.hist(results["human_seconds_log"], bins=50, alpha=0.7, label='Data')
    plt.xlabel("Average task time baseline (seconds, log2 scale)")
    plt.ylabel("Frequency")

    # Fit multiple distributions and find the best one
    distributions = ['cauchy', 'norm', 'lognorm', 'gamma', 'beta']
    fits = {}
    x = np.linspace(results["human_seconds_log"].min(), results["human_seconds_log"].max(), 1000)

    for dist_name in distributions:
        try:
            # Get the distribution from scipy.stats
            distribution = getattr(scipy_stats, dist_name)
            
            # Fit the distribution to our data
            params = distribution.fit(results["human_seconds_log"])
            
            # Calculate the PDF at our x points
            if dist_name == 'lognorm':
                # Special case for lognorm which has a different parameter order
                pdf = distribution.pdf(x, *params)
            else:
                pdf = distribution.pdf(x, *params)
            
            # Scale PDF to match histogram counts (total count × bin width)
            scaled_pdf = pdf * len(results["human_seconds_log"]) * bin_width
            
            # Calculate goodness of fit (using a simple sum of squared errors)
            hist_interp = np.interp(x, bin_centers, hist, left=0, right=0)
            pdf_interp = np.interp(bin_centers, x, scaled_pdf, left=0, right=0)
            sse = np.sum((hist - pdf_interp)**2)
            
            # Store results
            fits[dist_name] = {
                'params': params,
                'pdf': scaled_pdf,
                'sse': sse
            }
        except Exception as e:
            logging.warning(f"Failed to fit {dist_name} distribution: {e}")

    # Find the best fit based on sum of squared errors
    if fits:
        best_dist = min(fits, key=lambda k: fits[k]['sse'])
        
        # Plot the best fit
        plt.plot(x, fits[best_dist]['pdf'], 'r-', 
                label=f'{best_dist.capitalize()} fit (SSE={fits[best_dist]["sse"]:.2f})')
        
        # Plot Cauchy for comparison if it's not the best fit
        if best_dist != 'cauchy':
            plt.plot(x, fits['cauchy']['pdf'], 'g--', 
                    label=f'Cauchy fit (SSE={fits["cauchy"]["sse"]:.2f})')

    plt.legend(loc='upper right')
    plt.tight_layout()


    # Plot histogram of average human seconds per task id
    fig, ax = plt.subplots(figsize=(12, 6))
    df_filtered = df[df["task_source"] == "HCAST"]
    gdfs = df_filtered.groupby(["task_id"])
    results = gdfs["human_seconds"].mean().reset_index(drop=False)
    results["human_seconds_log"] = np.log2(results["human_seconds"])

    # Get the histogram data
    hist, bin_edges = np.histogram(results["human_seconds_log"], bins=50, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Plot histogram
    plt.hist(results["human_seconds_log"], bins=50, alpha=0.7, label='Data')
    plt.xlabel("Average task time baseline (seconds, log2 scale)")
    plt.ylabel("Frequency")

    # Focus on distributions that are most appropriate for this kind of data
    distributions = ['cauchy', 'norm', 'skewnorm', 'laplace', 't', 'logistic']
    fits = {}
    x = np.linspace(results["human_seconds_log"].min(), results["human_seconds_log"].max(), 1000)

    for dist_name in distributions:
        try:
            # Get the distribution from scipy.stats
            distribution = getattr(scipy_stats, dist_name)
            
            # Fit the distribution to our data with more robust settings
            if dist_name == 't':
                # For t-distribution, need to provide initial guess for df parameter
                params = distribution.fit(results["human_seconds_log"], f0=5)
            else:
                params = distribution.fit(results["human_seconds_log"])
            
            # Get parameter names for this distribution
            param_names = distribution.shapes.split(',') if distribution.shapes else []
            param_names = [name.strip() for name in param_names]
            
            # Add loc and scale which are always present
            all_param_names = param_names + ['loc', 'scale']
            param_values = dict(zip(all_param_names, params))
            
            # Calculate the PDF at our x points
            pdf = distribution.pdf(x, *params)
            
            # Scale PDF to match histogram counts
            scaled_pdf = pdf * len(results["human_seconds_log"]) * bin_width
            
            # Calculate goodness of fit (sum of squared errors)
            pdf_interp = np.interp(bin_centers, x, scaled_pdf, left=0, right=0)
            sse = np.sum((hist - pdf_interp)**2)
            
            # Also calculate AIC for model comparison
            loglik = np.sum(np.log(distribution.pdf(results["human_seconds_log"], *params) + 1e-10))
            k = len(params)  # Number of parameters
            aic = 2 * k - 2 * loglik
            
            # Store results with parameter details
            fits[dist_name] = {
                'params': params,
                'param_dict': param_values,
                'pdf': scaled_pdf,
                'sse': sse,
                'aic': aic
            }
            logging.info(f"Successfully fit {dist_name} distribution: SSE={sse:.2f}, AIC={aic:.2f}")
            
        except Exception as e:
            logging.warning(f"Failed to fit {dist_name} distribution: {str(e)}")

    # Find the best fit based on SSE and AIC
    if fits:
        best_sse_dist = min(fits, key=lambda k: fits[k]['sse'])
        best_aic_dist = min(fits, key=lambda k: fits[k]['aic'])
        
        # Plot the best SSE fit
        plt.plot(x, fits[best_sse_dist]['pdf'], 'r-', 
                label=f'{best_sse_dist.capitalize()} fit (SSE={fits[best_sse_dist]["sse"]:.2f})')
        
        # Plot the best AIC fit if different
        if best_aic_dist != best_sse_dist:
            plt.plot(x, fits[best_aic_dist]['pdf'], 'g--', 
                    label=f'{best_aic_dist.capitalize()} fit (AIC={fits[best_aic_dist]["aic"]:.2f})')
        
        # Print parameter details for the best fit
        logging.info(f"Best fit (SSE): {best_sse_dist} with parameters:")
        for name, value in fits[best_sse_dist]['param_dict'].items():
            logging.info(f"  {name}: {value:.4f}")
        
        # Plot Cauchy for comparison if it's not already the best fit
        if best_sse_dist != 'cauchy' and best_aic_dist != 'cauchy' and 'cauchy' in fits:
            plt.plot(x, fits['cauchy']['pdf'], 'b:', 
                    label=f'Cauchy fit (SSE={fits["cauchy"]["sse"]:.2f})')

    plt.legend(loc='upper right')
    plt.tight_layout()
    return fits

# compare_hist_fits(df)

plt.show()
