import prototype_phd.data_utils as data_utils
import prototype_phd.methods.egt as methods_egt
import prototype_phd.models as models
import prototype_phd.model_utils as model_utils
import prototype_phd.payoffs as payoffs
import prototype_phd.plot_utils as plot_utils
import prototype_phd.utils as utils

import logging
import matplotlib.pyplot as plt
import numpy as np

setup_args = {"log_path": "logs/ai_trust_media.log"}
simulation_id, current_commit, data_dir, plots_dir = data_utils.setup_project(setup_args)

Z = {"S4": 100, "S3": 100, "S2": 100, "S1": 100}
β = 0.1
sector_strategies = {"S4": [8, 9],
                     "S3": [5, 6],
                     "S2": [3, 4],
                     "S1": [1, 2], }
allowed_sectors = {"P4": ["S4"],
                   "P3": ["S3"],
                   "P2": ["S2"],
                   "P1": ["S1"], }
param_limits = {"cW": [0, 10],
                "pW": [0, 1],
                "bI": [0, 10],
                "cI": [0, 5],
                "bU": [0, 10],
                "bP": [0, 10],
                "cP": [0, 5],
                "Eps": [-5, 5],
                "u": [0, 10],
                "cR": [0, 5],
                "bR": [0, 10],
                "v": [0, 1],
                "b_fo": [0, 20],
                }
params_qmc = data_utils.generate_qmc_samples(param_limits, int(1e6))

params = {
          **models.build_ai_trust(Eps=[0.2, -0.1],
                           cR=[0.5, 5],
                           b_fo=[1, 5],
                           cW=[0.5, 5],
                           pW=0.5,
                           bI=[0.5, 5],
                           cI=np.arange(0, 10, 1)),
        #   **params_qmc,
          "dispatch-type": 'multiple-populations',
          "payoffs_key": "ai-trust-media-investigate-regulators",
          "β": β,
          "Z": Z,
          "allowed_sectors": allowed_sectors,
          "sector_strategies": sector_strategies,
          "simulation_id": simulation_id,
          "commit": current_commit,
          }

strategy_set:list[str]=["C-T-C-C", "C-T-C-D", "C-T-D-C", "C-T-D-D",
                    "C-N-C-C", "C-N-C-D", "C-N-D-C", "C-N-D-D",
                    "D-T-C-C", "D-T-C-D", "D-T-D-C", "D-T-D-D",
                    "D-N-C-C", "D-N-C-D", "D-N-D-C", "D-N-D-D"]
params["strategy_set"] = strategy_set
# Check the size of the created arrays using:
params_shapes = {k: params[k].shape for k,v in params.items() if isinstance(v, np.ndarray)}
logging.info(f"Parameter shapes: {params_shapes}")

# Run and save simulations
results = utils.thread_macro(params,
                       model_utils.create_profiles,
                       model_utils.apply_profile_filters,
                       payoffs.build_payoffs,
                       methods_egt.build_transition_matrix,
                       methods_egt.find_ergodic_distribution,
                       )

df = utils.thread_macro(results,
                  data_utils.results_to_dataframe_egt,
                  data_utils.process_ai_trust_dataframe,
                  )

data_utils.save_data({"params": params,
                      "results": df},
                     data_dir=data_dir)

model_name = params["payoffs_key"]
# Add workaround for better labels for frquency columns
# print(results["recurrent_states"])
results["recurrent_states"] = results["strategy_set"]
result_sums = np.sum(results['ergodic'], axis=-1)
# Test that all entries in result_sums are close to 1
assert np.allclose(result_sums, 1, atol=1e-10)
# Workaround to ensure we can easily slice the dataframe by the Eps column
df["Eps"] = np.round(df["Eps"], 2)
# Create columns for frequency of each player cooperating
df, _strat_set = data_utils.compute_strategy_frequencies(df, results["recurrent_states"])
freq_cols = [col for col in df.columns if col.endswith('frequency')]
# Rename columns for better labels
old_cols = ["P1_strat_C_frequency",
             "P2_strat_C_frequency",
             "P3_strat_T_frequency",
             "P4_strat_C_frequency"]
new_cols = ["Regulator_Cooperates_frequency",
            "Developer_Cooperates_frequency",
            "User_Trusts_frequency",
            "Media_Cooperates_frequency"]
rename_dict = {old: new for old, new in zip(old_cols, new_cols)}
df = df.rename(columns=rename_dict)

def plot_lines_vary_cI(plot_df, plot_cols):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Marker options for distinguishing cW values
    marker_options = ['o', 's', '^', 'd', 'v']
    cW_values = sorted(plot_df['cW'].unique())
    cW_to_marker = {cW: marker_options[i % len(marker_options)] for i, cW in enumerate(cW_values)}

    for i, col in enumerate(plot_cols):
        ax = axs[i//2, i%2]
        # Group the data by the two variables that will drive line style.
        groups = plot_df.groupby(['bI', 'cW'])

        for idx, ((bI_val, cW_val), group) in enumerate(groups):
            # Use Dark2 categorical colormap with discrete indices
            # color = plt.cm.Dark2(idx % 8)
            color = plt.cm.tab10(idx % 10)
            marker = cW_to_marker[cW_val]

            # Plot a line for this group.
            ax.plot(group['cI'], group[col],
                    linestyle='-', marker=marker, color=color,
                    label=f"bI: {bI_val:.2f}, cW: {cW_val:.2f}")

        ax.set_title("", fontsize=18)
        ax.set_xlabel('cI', fontsize=18)
        ax.set_ylabel(col.replace("_", " "), fontsize=18)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylim(0, 1.05)

        # Build a legend with unique labels.
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=12, title='bI, cW', title_fontsize=12)
    return fig, axs

def plot_heatmaps_bI_cW(plot_df, plot_cols):
    
    # Create a figure with 4 subplots, each one is a heatmap for a different frequency column
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    if len(plot_df) > 0:
        for i, col in enumerate(plot_cols):
            table = plot_df.pivot_table(index='cW', columns='bI', values=col)
            ax = axs[i//2, i%2]
            _fig, _ax, im = plot_utils.plot_heatmap(table,
                                    xlabel='Benefit of Investigators, bI',
                                    ylabel='Reputational Cost, cW',
                                    zlabel=col,
                                    cmap='RdBu_r',
                                    figure_object=[fig, ax],
                                    set_colorbar=False
                                    )
            # Remove individual colorbars or ticks if desired:
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.set_title(col.split('_')[0])
    # Create a shared colorbar for the entire figure.
    # Note: We use the last image (img_for_colorbar). All images are assumed to be on a similar scale.
    fig.subplots_adjust(bottom=0.1)  # Adjust bottom to make space for the colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.04)
    cbar.set_label('Cooperation Frequency')

    return fig, axs

def plot_histograms_coop_frequency(plot_df, plot_cols):
    
    # Create subplot histograms for the cooperation frequencies of each player across all data
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for i, col in enumerate(new_cols):
        ax = axs[i//2, i%2]
        plot_hist = df[col].plot.hist(ax=ax, bins=20, range=(0, 1), alpha=0.5)
        ax.set_title(col)
    return fig, axs

if len(df) > 0:
    plot_limit = 200 # Limit the number of plots to avoid saving too many figures
    plots = {}
    plot_cols = new_cols
    # Plot line plots as we vary cI
    group_vars = ['cR', 'Eps', 'b_fo']
    plot_df_groups = df.groupby(group_vars)
    for comb, group in plot_df_groups:
        if len(plot_df_groups) > plot_limit:
            logging.info(f" Number of groups = {len(plot_df_groups)}. Too many groups to plot, skipping.")
            break
        # Construct the title by zipping group_vars with comb
        title = ', '.join(f"{var}={val}" for var, val in zip(group_vars, comb))
        # title += f"\n{model_name}"
        fig, axs = plot_lines_vary_cI(group, plot_cols)
        fig.suptitle(title, fontsize=24)
        for ax in axs.flat:
            ax.set_ylim(0, 1.05)
        
        # Create a key using the grouping variables
        key = "fig_vary_cI_" + '_'.join(f"{var}_{val}" for var, val in zip(group_vars, comb))
        key = f"{key}_{model_name}_{simulation_id}_{current_commit}"
        plots[key] = fig
    
    # # Plot heatmaps for bI and cW
    # group_vars = ['cR', 'Eps', 'b_fo', 'cI']
    # plot_df_groups = df.groupby(group_vars)
    # for comb, group in plot_df_groups:
    #     if len(plot_df_groups) > plot_limit:
    #         logging.info(f" Number of groups = {len(plot_df_groups)}. Too many groups to plot, skipping.")
    #         break
    #     # Construct the title by zipping group_vars with comb
    #     title = ', '.join(f"{var}={val}" for var, val in zip(group_vars, comb))
    #     title += f"\n{model_name}"
    #     fig, axs = plot_heatmaps_bI_cW(group, plot_cols)
    #     fig.suptitle(title, fontsize=24)
    #     # Create a key using the grouping variables
    #     key = "fig_heatmap_bI_cW_" + '_'.join(f"{var}_{val}" for var, val in zip(group_vars, comb))
    #     key = f"{key}_{model_name}_{simulation_id}_{current_commit}"
    #     plots[key] = fig
    
    # # Plot histograms for cooperation frequencies
    # fig, axs = plot_histograms_coop_frequency(df, plot_cols)
    # plots[f"fig_histograms_{model_name}_{simulation_id}_{current_commit}"] = fig
    data_utils.save_plots(plots, plots_dir=plots_dir)

