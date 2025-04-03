import prototype_phd.data_utils as data_utils
from prototype_phd.methods.mcdev import (
    PlotConfig,
    ScenarioConfig,
    plot_allocations,
    scenario_constant,
    scenario_increasing,
    scenario_diminishing,
)
import numpy as np

project_args = data_utils.setup_project(log_path='logs/mcdev.log')
sim_id, commit, data_dir, plots_dir = project_args
K = 20
# Define scenarios
scenarios = [
    ScenarioConfig(
        scenario_name="Constant Increase",
        scenario_func=scenario_constant,
        K=K
    ),
    ScenarioConfig(
        scenario_name="Increasing Increase",
        scenario_func=scenario_increasing,
        K=K
    ),
    ScenarioConfig(
        scenario_name="Diminishing Increase",
        scenario_func=scenario_diminishing,
        K=K
    ),
]

alpha_values = [0.0, 0.5, 0.9]
B_values = list(np.linspace(10, 100, 10))

# psi=1, gamma=1
psi = np.ones(K)
gamma = np.ones(K)

plots = {}  # Dictionary to collect figure objects
for scenario in scenarios:
    for alpha in alpha_values:
        pc_direct = PlotConfig(
            scenario=scenario,
            alpha=alpha,
            B_values=B_values,
            psi=psi,
            gamma=gamma
        )
        fig = plot_allocations(pc_direct)
        key = f"{scenario.scenario_name.replace(' ', '_')}_alpha_{alpha}"
        key += f"sim_id_{sim_id}.png"
        plots[key] = fig

data_utils.save_plots(plots, plots_dir=f"{plots_dir}/mcdev")
