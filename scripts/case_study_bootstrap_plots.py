import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prototype_phd.data_utils as data_utils
import seaborn as sns
import tqdm as tqdm

# Note: We specify sim_id in setup_project so that we can carry on the work
# from a previous simulation. This is useful for iterating on plotting code
# without having to re-run the entire simulation.
# TODO: In future, consider DVC instead for versioning data and plots.
# sim_id = "administering_stalwart_chat_498d60cb"
# sim_id = "fragrance_scorecard_newton_aea31b12"
# sim_id = "Anglicans_engraved_alliance_4978e5bf"
sim_id = None
if sim_id is None:
    sim_id = data_utils.get_latest_sim_id("data/detection_rates_case_study/sim_tracker.csv")
    print(f"Latest simulation ID: {sim_id}")
setup_args = {"log_path": "logs/detection_rates_case_study.log",
              "simulation_id": sim_id,
              "data_dir_root": "data/detection_rates_case_study",
              "plots_dir_root": "plots/detection_rates_case_study",}
sim_id, commit, data_dir, plots_dir = data_utils.setup_project(**setup_args)

logging.info(f"Simulation ID: {sim_id}")
logging.info(f"Commit: {commit}")

# Load external data
external_data_dir = data_dir
file_path = f"{external_data_dir}/df_bootstrap.csv"
df_bootstrap = pd.read_csv(file_path)
logging.info(f"External data columns: {df_bootstrap.columns}")

def plot_distributions(df: pd.DataFrame) -> pd.DataFrame:
    
    # NOTE: Beware irregularities caused by the filter below.
    # To see threshold distribution better, filter any rows with coeff_1 within tol of 0
    tol = 1e-1
    df = df[df["coeff_1"].abs() > tol]
    df = df[df["convergence"]]

    # Visualize the bootstrap distributions for each model and budget value
    plots = {}
    plot_group_vars = ["model"]
    for group, gdf in tqdm.tqdm(df_bootstrap.groupby(plot_group_vars)):
        # select numeric metric columns (exclude grouping vars and Budget)
        metric_cols = [
            c for c in gdf.select_dtypes(include=[np.number]).columns
            if c not in plot_group_vars + ["Budget", "warning"]
        ]
        print("Metric columns:", metric_cols)
        for col in tqdm.tqdm(metric_cols):
            # faceted histogram by Budget
            g = sns.displot(
                data=gdf,
                x=col,
                col="Budget",
                col_wrap=4,
                height=3,
                aspect=1.5,
                bins=30,
                facet_kws={"sharey": False},
            )
            g.set_axis_labels(col, "Count")
            g.set_titles("Budget = {col_name}")
            # build a safe group string
            group_string = " ".join(
                f"{var}={val}".replace(" ", "_")
                for var, val in zip(plot_group_vars, group)
            )
            g.figure.suptitle(f"Bootstrap distribution for {col}, group: {group_string}", y=1.02)
            plt.tight_layout()
            plot_key = f"{'_'.join(map(str, group))}_{col}"
            plots[plot_key] = g.figure
    return plots

def plot_bias_curves(df: pd.DataFrame, estimator_var="threshold") -> pd.DataFrame:
    """
    Process the bootstrap data to create bias curves.
    """
    # Scan the estimator columns to find the ones that start with "estimate_"
    # and then order the columns by the suffixes.
    # We assume the estimator columns are named "estimate_1", "estimate_2", etc.
    # For each row, we want to find the highest suffix for which the value is above 0.5
    
    plots = {}

    df_thresh = df.groupby(
        ["model", "Budget_fraction"],
        observed=True, dropna=True
    )[estimator_var] \
        .mean() \
        .reset_index(name=f"mean_{estimator_var}")

    # For “reported threshold,” pick the highest Budget per model:
    reported_thresh = df_thresh.groupby("model", observed=True, dropna=True, as_index=False).apply(
        lambda g: g.loc[g["Budget_fraction"].idxmax(), ["model", f"mean_{estimator_var}"]]
    ).rename(columns={f"mean_{estimator_var}": f"reported_{estimator_var}"})

    # Merge this reported_threshold onto df_thresh so you can plot them together:
    df_thresh = pd.merge(df_thresh, reported_thresh, on="model", how="left")

    # First exclude outlier threshold values
    # (e.g. any negative values or values greater than 20)
    ub = 20 # upper bound for threshold, not tasks are above 2**20 seconds in our data.
    # Note: this is a bit arbitrary, but we can adjust it later if needed.
    df_thresh = df_thresh[(df_thresh["mean_threshold"] >= 0) & (df_thresh["mean_threshold"] <= ub)]

    import math
    # matplotlib faceted bar charts by Budget_fraction
    budgets = sorted(df_thresh["Budget_fraction"].unique())
    n = len(budgets)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(4*ncols, 3*nrows),
                              sharex=True, sharey=True)
    axes = axes.flatten()
    cmap = plt.cm.viridis
    for i, bf in enumerate(budgets):
        ax2 = axes[i]
        sub = df_thresh[df_thresh["Budget_fraction"] == bf]
        ax2.bar(sub[f"reported_{estimator_var}"],
                sub[f"mean_{estimator_var}"],
                color=cmap(bf))
        # identity line
        lims = [
            min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
            max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ]
        ax2.plot(lims, lims, "--", color="gray")
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_title(f"Budget Fraction = {bf:.2f}")
        ax2.set_xlabel(f"Reported {estimator_var}")
        ax2.set_ylabel(f"Bootstrap mean {estimator_var}")
    # hide unused axes
    for ax2 in axes[len(budgets):]:
        ax2.set_visible(False)
    fig2.suptitle(f"Reported vs. Bootstrap Mean {estimator_var}\nby Budget Fraction", y=1.02)
    plt.tight_layout()
    plots[f"reported_vs_bootstrap_{estimator_var}_by_fraction"] = fig2


    plot_group_vars = ["model"]
    for group, gdf in tqdm.tqdm(df_thresh.groupby(plot_group_vars, observed=True)):
        sub = gdf
        fig = plt.figure()
        plt.plot(sub["Budget_fraction"], sub[f"mean_{estimator_var}"], marker='o', label=f"Bootstrap mean {estimator_var}")
        plt.axhline(y=sub[f"reported_{estimator_var}"].iloc[0], color='r', linestyle='--', label=f"Reported {estimator_var} (Highest Budget)")
        plt.title(f"{estimator_var} vs. budget (model: {group})")
        plt.xlabel("Budget fraction")
        plt.ylabel(estimator_var)
        plt.legend()
        plt.tight_layout()

        # build a safe group string
        group_string = " ".join(
            f"{var}={val}".replace(" ", "_")
            for var, val in zip(plot_group_vars, group)
        )
        plot_key = f"{group_string}_{estimator_var}_bias"
        plots[plot_key] = fig
    
    # Debug bootstrap distributions
    # plots2 = plot_distributions(df)
    # plots.update(plots2)
    
    return plots

def plot_detection_rates(df: pd.DataFrame) -> pd.DataFrame:
    
    # Also, plot how likelihood of sample threshold > critical_threshold varies
    # with reported threshold for different values of budget


    critical_threshold = 9  # pick your critical threshold


    # After loading df_bootstrap and filtering, you can group by (model, Budget) to compute mean threshold:
    df_thresh = df.groupby(["model", "Budget_fraction"], observed=True, dropna=True)["threshold"].mean().reset_index(name="mean_threshold")

    # For “reported threshold,” pick the highest Budget per model:
    reported_thresh = df_thresh.groupby("model", observed=True, dropna=True, as_index=False).apply(
        lambda g: g.loc[g["Budget_fraction"].idxmax(), ["model", "mean_threshold"]]
    ).rename(columns={"mean_threshold": "reported_threshold"})

    # Merge this reported_threshold onto df_thresh so you can plot them together:
    df_thresh = pd.merge(df_thresh, reported_thresh, on="model", how="left")

    # Compute probability that each bootstrap threshold exceeds critical_threshold
    df_prob = df_bootstrap.groupby(["model", "Budget"], observed=True, dropna=True).apply(
        lambda g: (g["threshold"] > critical_threshold).mean()
    ).reset_index(name="prob_over_crit")

    # Merge in the reported_threshold from df_thresh (which we generated earlier)
    df_prob = pd.merge(
        df_prob,
        df_thresh[["model", "Budget", "reported_threshold"]].drop_duplicates(),
        on=["model", "Budget"],
        how="left"
    )

    # Plot prob_over_crit vs. Budget, with reported_threshold on a second axis
    plot_group_vars = ["model"]
    for group, gdf in tqdm.tqdm(df_prob.groupby(plot_group_vars, observed=True)):
        fig, ax1 = plt.subplots()
        
        ax1.plot(
            gdf["Budget"],
            gdf["prob_over_crit"],
            marker='o',
            color='b',
            label=f"P(threshold > {critical_threshold})"
        )
        ax1.set_xlabel("Budget")
        ax1.set_ylabel("Probability Over Critical Threshold", color='b')
        
        ax2 = ax1.twinx()
        ax2.plot(
            gdf["Budget"],
            gdf["reported_threshold"],
            marker='o',
            color='r',
            label="Reported Threshold"
        )
        ax2.set_ylabel("Reported Threshold", color='r')
        
        plt.title(f"Model: {group} — Probability(Threshold > {critical_threshold}) vs. Reported Threshold")
        fig.tight_layout()

        # Optionally save the figure in plots dict
        plot_key = f"{group}_prob_thresh"
        plots[plot_key] = fig
    return plots

# plots1 = plot_distributions(df_bootstrap)

df_bootstrap["Budget_fraction"] = np.round(df_bootstrap["Budget_fraction"], 2)
plots2 = plot_bias_curves(df_bootstrap)

# plots3 = plot_detection_rates(df_bootstrap)

plots = {
    # **plots,
         **plots2}

data_utils.save_plots(plots, plots_dir=f"{plots_dir}/case_study_plots")

# TODO: Investigate why coeff_1 can be both positive and negative
# - Is it noise in the logistic regression fitting?
# - Is it noise due to the bootstrap sampling?
# Note: I didn't quite expect to see such a large spread in coeff_1
# TODO: Investigate whether most regressions converged
# TODO: Improve x-axis labels for the threshold plot
# TODO: Annoate threshold plot with the expected threshold (and the reported threshold)


