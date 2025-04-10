import prototype_phd.data_utils as data_utils
import prototype_phd.methods.mcdev as mcdev
import prototype_phd.utils as utils
import numpy as np

project_args = data_utils.setup_project(log_path='logs/mcdev.log')
sim_id, commit, data_dir, plots_dir = project_args

def build_scenarios():
    K = 20
    # Define scenarios
    scenario_config = mcdev.ScenarioConfig(
            scenario_name="Exponential Increase",
            scenario_func=mcdev.scenario_exponential,
            K=K)
    k_vec = np.arange(1, K + 1)
    p = scenario_config.scenario_func(k_vec)
    # alpha_values = [-2.0, 0.0, 0.5, 0.9]
    alpha_values = [0.0]
    B_values = list(np.linspace(10, 100, 5))
    B_values.append(500)
    B_values.append(1000)
    B_values.append(10000)
    B_values.append(100000)

    psi = np.ones(K)
    gamma = np.ones(K)
    gamma = 5 / (1 + np.exp(-0.5 * (np.arange(K) - K/2)))
    gamma = np.linspace(0.25, 4.75, K)
    # Have psi and gamma increase at a rate slower than the prices
    # at least in the linear case they increase from 1 to 5 there
    # Remember, the shadow price doesn't control the relative allocations,
    # only psi, price, gamma, and alpha do. For gamma large enough, we
    # might see that some items once allocated get more than those that
    # were previously allocated to.
    # psi = np.linspace(1, 3, K)
    psi = mcdev.scenario_exponential(np.arange(K), 1, 0.5)
    # gamma = np.exp(np.linspace(1, 2, K))

    variable_parameters = {
        "B": B_values,
        "p": [p],
        "psi": [psi],
        "gamma": [gamma],
        "alpha": alpha_values,
        "scenario_config": [scenario_config],
    }
    configs = [mcdev.DemandConfig(**d)
               for d in utils.dict_list(variable_parameters)]
    return configs

# Build scenarios
scenarios = build_scenarios()
# Run the model
df = mcdev.compute_allocations(scenarios)
# Process data
df["Cumulative_Demand"] = df.groupby("Budget")["Demand"].cumsum()
df["Cumulative_Expenditure"] = df.groupby("Budget")["Expenditure"].cumsum()
# Create plots
group_vars = ["Alpha", "Scenario"]
gdfs = df.groupby(group_vars)
plots = {}
for plot_group, gdf_index in gdfs.groups.items():
    plot_group = dict(zip(group_vars, plot_group))
    gdf = df.loc[gdf_index]
    for y_var in ["Expenditure", "Demand", "Cumulative_Demand", "Cumulative_Expenditure"]:
        pc = mcdev.PlotConfig(
            plot_group=plot_group,
            df=gdf,
            y_var=y_var,
        )
        fig = mcdev.plot_allocations(pc)
        group_string = [f"{group_var}_{group_val}".replace(" ", "_")
                        for group_var, group_val in plot_group.items()]
        group_string = "_".join(group_string)
        key = f"{y_var}_{group_string}_sim_id_{sim_id}"
        plots[key] = fig
data_utils.save_plots(plots, plots_dir=f"{plots_dir}/mcdev")
