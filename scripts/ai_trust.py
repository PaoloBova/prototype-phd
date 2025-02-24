import prototype_phd.data_utils as data_utils
import prototype_phd.methods.egt as methods_egt
import prototype_phd.models as models
import prototype_phd.model_utils as model_utils
import prototype_phd.payoffs as payoffs
import prototype_phd.plot_utils as plot_utils
import prototype_phd.utils as utils

simulation_id, current_commit, data_dir, plots_dir = data_utils.setup_project()

params = {**models.build_ai_trust(Eps={"start": -1, "stop": 1, "step": 0.1},
                                  β=1,
                                  cR=0.5,
                                  strategy_set=["N-C-C", "N-C-D", "N-D-C", "N-D-D", "CT-C-C", "CT-C-D", "CT-D-C", "CT-D-D"],
                           ),
         "simulation_id": simulation_id,
          "commit": current_commit,
         "dispatch-type": 'multiple-populations',
          "payoffs_key": "ai-trust-v1",
          "Z": {"S3": 100, "S2": 100, "S1": 100},
          "allowed_sectors": {"P3": ["S3"],
                              "P2": ["S2"],
                              "P1": ["S1"], },
          "sector_strategies": {"S3": [6, 7],
                                "S2": [3, 4],
                                "S1": [1, 2], },
          }

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

data_utils.save_data({
  # "params": params,
                      "results": df},
                     data_dir=data_dir)

# Plot results
plot1 = plot_utils.plot_strategy_distribution(df[df["Eps"] == -0.1],
                           results['strategy_set'],
                           x="b_fo",
                           x_label="b_fo",
                           thresholds=None,
                           )
plot2 = plot_utils.plot_strategy_distribution(df[df["Eps"] == 0.2],
                           results['strategy_set'],
                           x="b_fo",
                           x_label="b_fo",
                           thresholds=None,
                           )

# Save plots and animations
plots = {f"fig1a_{simulation_id}_{current_commit}": plot1,
         f"fig1b_{simulation_id}_{current_commit}": plot2,
      }
data_utils.save_plots(plots, plots_dir=plots_dir)
