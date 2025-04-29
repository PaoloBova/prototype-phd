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
file_path = f"{external_data_dir}/df_analytical.csv"
df_analytical = pd.read_csv(file_path)
logging.info(f"External data columns: {df_analytical.columns}")

# Visualize the bootstrap distributions for each model and budget value
plots = {}
plot_group_vars = ["model"]
for group, gdf in tqdm.tqdm(df_analytical.groupby(plot_group_vars)):
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
        # build a safe group string
        group_string = " ".join(
            f"{var}={val}".replace(" ", "_")
            for var, val in zip(plot_group_vars, group)
        )
        g.set_titles("Budget = {col_name}")
        g.figure.suptitle(f"Bootstrap distribution for {col}, group: {group_string}", y=1.02)
        plt.tight_layout()
        plot_key = f"{'_'.join(map(str, group))}_{col}"
        plots[plot_key] = g.figure

data_utils.save_plots(plots, plots_dir=f"{plots_dir}/case_study_plots")

