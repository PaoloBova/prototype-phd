import sys
import os

import prototype_phd.data_utils as data_utils
import prototype_phd.methods.egt as methods_egt
import prototype_phd.models as models
import prototype_phd.model_utils as model_utils
import prototype_phd.payoffs as payoffs
import prototype_phd.plot_utils as plot_utils
import prototype_phd.utils as utils

import json
import numpy
import pandas

def load_fairgame_data(params):
    """Load all fairgame data from the specified directories.
    
    We have external data in the form of CSV files and internal data in the form of JSON files."""
    
    data_dir = params["data_dir"]
    sims = params["sims"]
    external_data_dir = params["external_data_dir"]
    
    params = []
    configs = []
    config_dfs = []
    for sim in sims:

        # Load all fairgame configs
        
        fairgame_configs = []
        for file in os.listdir(f"{data_dir}/fairgame_configs/{sim}"):
            if file.startswith("FAIRGAME"):
                config_index = file.split("_")[-1].split(".")[0]
                version_index = file.split("_")[-2]
                with open(f"{data_dir}/fairgame_configs/{sim}/{file}", 'r') as f:
                    data = json.load(f)
                    data["config_index"] = int(config_index)
                    data["version_index"] = version_index
                    data["simulation_id"] = sim
                    fairgame_configs.append(data)
                    configs.append(data)

        # Collect the configs for the current simulation into a single dataframe  
        config_df = pandas.concat([data_utils.json_to_df(config) for config in fairgame_configs])
        config_dfs.append(config_df)
        
        # Load params.json file from data_dir
        
        with open(f"{data_dir}/fairgame_configs/{sim}/_params.json", 'r') as f:
            data = json.load(f)
            # Give params dictionaries a config_index field to help with merging 
            # alternate results data later
            data["config_index"] = list(range(len(data["Eps"])))
            params.append(data)

    params_df = pandas.concat([data_utils.json_to_df(p) for p in params]).reset_index()
    # Confirm that the config index we get now is the same as if we do the following:
    params_df["config_index"] = params_df["index"].astype(int)

    # Concatenate all the fairgame config dataframes into a single dataframe
    configs_df = pandas.concat(config_dfs)

    # Load results csv files from external_data directory
    # Iterate over all csv files in the directory and concat them into one dataframe
    # Typically, the external data has all the results together for all simulations
    results = []
    for file in os.listdir(external_data_dir):
        if file.endswith(".csv"):
            df = pandas.read_csv(f"{external_data_dir}/{file}", sep=";")
            filename = file.split(".")[0]
            df["filename"] = filename
            results.append(df)
    results_df = pandas.concat(results)

    return {"params": params,
            "params_df": params_df,
            "configs": configs,
            "configs_df": configs_df,
            "results": results,
            "results_df": results_df}

def results_to_tidy_dataframe(df):
    # List game-level columns
    game_fields = ['game_id', 'language', 'n_rounds_is_known', 'max_rounds', 'agents_communicate', 'played_rounds', 'filename']
    
    # Agent-level fields where value is not round-specific
    constant_agent_fields = ['name', 'llm', 'personality', 'knows_opponent_with_prob']
    
    # Agent-level fields that are lists (one element per round)
    list_agent_fields = ['strategies', 'scores', 'messages']
    
    # Calculate number of agents
    n_players = len([col for col in df.columns if col.startswith('agent') and col.endswith('_name')])
    
    # Make sure lists are not strings:
    # Some columns of results_df have lists as values but they are currently represented as strings.
    # We need to convert them to lists
    for i in range(1, n_players+1):
        for col_name in [f"agent{i}_scores", f"agent{i}_strategies", f"agent{i}_messages"]:
            if isinstance(results_df[col_name].iloc[0], str):
                results_df[col_name] = results_df[col_name].apply(lambda x: eval(x))
    
    # Verify that the dataframe stores round lists consistently
    # All round-based columns must share the same length equal to rounds_played
    # n_rounds = results_df["played_rounds"].unique()[0]
    # for i in range(1, n_players+1):
    #     for col_name in [f"agent{i}_scores", f"agent{i}_strategies", f"agent{i}_messages"]:
    #         # print(f"Checking {col_name}")
    #         # print(results_df[col_name].apply(len).unique())
    #         assert len(results_df[col_name].apply(len).unique()) == 1
    #         if results_df[col_name].apply(len).unique()[0] == 0:
    #             continue
    #         assert results_df[col_name].apply(len).unique()[0] == n_rounds

    
    tidy_rows = []
    
    for _, row in df.iterrows():
        rounds = row['played_rounds']  # assuming this tells the number of rounds played
        base_info = {field: row[field] for field in game_fields}
        
        # Process each agent
        for agent in range(1, n_players+1):
            # Read the constant fields for the agent.
            agent_constant = {
                f'agent_{field}': row[f'agent{agent}_{field}'] for field in constant_agent_fields
            }
            # For each round create a separate row.
            for rnd in range(rounds):
                row_dict = base_info.copy()
                row_dict['round'] = rnd + 1
                row_dict['agent'] = agent
                # Insert the constant agent fields.
                row_dict.update(agent_constant)
                # For each list field, check if this entry is a list; if yes, get the round-specific value.
                for field in list_agent_fields:
                    col_name = f'agent{agent}_{field}'
                    value = row[col_name]
                    # If the value is a list and the round exists, take that round's value; otherwise, use the entire value.
                    if isinstance(value, list) and len(value) > rnd:
                        row_dict[f'agent_{field}'] = value[rnd]
                    else:
                        row_dict[f'agent_{field}'] = value
                tidy_rows.append(row_dict)
    
    return pandas.DataFrame(tidy_rows)

def check_scores_consistency(df):
    """
    For each game and round (group of n_agent rows), this function:
      1. Extracts the strategies chosen by the three agents.
      2. Iterates over the available payoffMatrix_combinations_combination columns until one
         of them matches the strategies observed.
      3. For the matching combination, uses the weight identifier to look up the expected score and compare it
         with the agent_scores column.
      
    Returns a list of error messages for mismatches or missing combinations.
    
    Check that the scores for each agent match the scores they should receive
    according to the weight associated with the strategy profile played.

    """
    # Names of the combination columns (adjust if more or fewer)
    
    comb_cols = [col for col in df.columns if col.startswith('payoffMatrix_combinations_combination')]
    # Build a mapping for weight columns; assuming weight ids are integers 1-24
    weight_cols = [col for col in df.columns if col.startswith('payoffMatrix_weights_weight_')]
    weight_cols = {f"weight_{i}": col for i, col in enumerate(weight_cols, 1)}
    strategy_cols = [col for col in df.columns if col.startswith('payoffMatrix_strategies')]
    # Count number of agents (assumes all rows have the same number of agents)
    n_players = df.agent.nunique()
    errors = []

    # Group by game id and round number so that each group has the three agents
    grouped = df.groupby(['game_id', 'round'])
    
    for (game_id, round_num), group in grouped:
        # Build a dictionary mapping agent index (assumed to be 1,2,3) to their strategy value.
        strategies = {}
        for _, row in group.iterrows():
            agent_idx = row['agent']
            agent_strategy_name = row['agent_strategies']
            # Find agent_strategy that has matching agent_strategy_name in the strategy columns
            agent_strategy = None
            for col in strategy_cols:
                if row[col] == agent_strategy_name:
                    agent_strategy_col_name = col
                    # print("agent_strategy_col_name", agent_strategy_col_name)
                    agent_strategy = agent_strategy_col_name.split('_')[-1]
                    break
            strategies[agent_idx] = agent_strategy
        
        # We assume that the payoffMatrix combination columns are identical within a group,
        # so we can take the first row.
        first = group.iloc[0]
        combination_found = False
        
        for comb_col in comb_cols:
            comb_list = first[comb_col]
            comb_list = eval(comb_list)
            # Expecting comb_list to be something like:
            # [ (expected_strategy_agent1, weight_id1),
            #   (expected_strategy_agent2, weight_id2),
            #   (expected_strategy_agent3, weight_id3) ]
            # print("strategies", strategies, type(strategies))
            if isinstance(comb_list, list) and len(comb_list) == n_players:
                # Check each agent's strategy against the combination.
                # print("comb_list", comb_list, type(comb_list))
                # print("strategies: ", strategies)
                if all(comb_list[i][0] == strategies.get(i+1)
                       for i in range(n_players)):
                    # Found the matching combination.
                    combination_found = True
                    # print("Combination found")
                    # Now check each agent's score.
                    for _, row in group.iterrows():
                        agent_idx = row['agent']
                        # Get the expected weight id using the appropriate tuple in comb_list.
                        expected_weight_id = comb_list[agent_idx - 1][1]
                        expected_weight_ids = [comb_list[i][1] for i in range(n_players)]
                        expected_weights = [weight_cols.get(weight_id) for weight_id in expected_weight_ids]
                        weight_col_name = weight_cols.get(expected_weight_id)
                        # print("Details: ", agent_idx, expected_weight_id, weight_col_name)
                        if weight_col_name is None:
                            errors.append((game_id, round_num, agent_idx,
                                           f"Weight id {expected_weight_id} not found."))
                        else:
                            expected_score = row[weight_col_name]
                            actual_score = row['agent_scores']
                            if expected_score != actual_score:
                                error_msg = f"Score mismatch: expected {expected_score}, got {actual_score}."
                                # Check if the correct score was given to one of the agents in that combination.
                                if expected_score in expected_weights:
                                    error_msg += f" __Score was given to another agent."
                                # Check if the correct score was found in another weight column.
                                for col in weight_cols.values():
                                    if row[col] == expected_score:
                                        error_msg += f" __Score found in another weight column: {col}."
                                errors.append((game_id, round_num, agent_idx, error_msg))
                            expected_score = row[weight_col_name]
                    break
        if not combination_found:
            errors.append((game_id, round_num, None,
                           "No matching strategy combination found for this game-round."))
    return errors

def extract_version_index(filename):
    """Extract all three from the filename column"""
    parts = filename.split('_')
    version = parts[-4]
    config_index = int(parts[-3])
    replication_index = int(parts[-1])
    return version, config_index, replication_index

def add_indices_to_df(df, sim_mapping):
    """Add indices to results df.
    
    We can use the filename column to identify the params.json file for each game
    Each params_df row has a simulation_id and index. We know the mapping from
    v1, v2 to the relevant simulation id.
    The filename column in df_tidy contains both the version number and the index
    number, as well as the replication number.
    """

    df["simulation_id"] = df['filename'].apply(lambda x: sim_mapping[extract_version_index(x)[0]])
    df["config_index"] = df['filename'].apply(lambda x: extract_version_index(x)[1])
    df["replication_index"] = df['filename'].apply(lambda x: extract_version_index(x)[2])

    df = df.merge(params_df, on=["simulation_id", "config_index"])
    df = df.merge(configs_df, on=["simulation_id", "config_index"])
    df["game_id"] = df["simulation_id"] + "_" + df["config_index"].astype(str) + "_" + df["replication_index"].astype(str)

    return df

def consistency_check1(df):
    """Check that the scores for each agent match the scores they should receive according to the weight
    associated with the strategy profile played."""
    mismatches = check_scores_consistency(df)
    if mismatches:
        print("Number of mismatches found:", len(mismatches))
        for err in mismatches:
            game, rnd, agent, msg = err
            agent_str = f"Agent {agent}" if agent is not None else "Group"
            print(f"Game {game}, Round {rnd}, {agent_str}: {msg}")
    # else:
    #     print("All agent_scores are consistent with the payoff matrix weights.")
    return None

def consistency_check2(configs_df, sim_df):
    """A second consistency check for payoff matrix weights, comparing the weights
    in configs_df with the weights in sim_df."""
    weight_cols = [col for col in configs_df.columns if col.startswith('payoffMatrix_weights_weight_')]
    for col in weight_cols:
        if col.endswith("_compare"):
            continue
        if f"{col}_compare" not in configs_df.columns:
            # Assume the consistency check is not necessary
            continue
        x1 = configs_df.sort_values(by=["simulation_id","config_index"])[col]
        x2 = sim_df.sort_values(by=["simulation_id","config_index"])[f"{col}_compare"]
        x = numpy.isclose(x1,x2)
        n_mismatches = len(x) - sum(x)
        if n_mismatches > 0:
            print(x)
            print(f"Column {col} has {n_mismatches} mismatches.")
            print(f"Expected values: {x1}")
            print(f"Actual values: {x2}")
    return None

def df_to_observed_data(df_tidy, params_df, strategy_id_mapping):
    """Derives observed data on strategy profile frequencies from the tidy dataframe.
    
    Notes:
    - Assumes all groups have the same number of agents."""
    # ==================================================
    # Part 0: Create strategy_id columns with the given mapping
    # ==================================================
    df_tidy["agent_strategy_id"] = df_tidy.apply(
        lambda x: strategy_id_mapping[x["agent_name"]][x["agent_strategies"]], axis=1)
    # ==================================================
    # Part 1: Compute Observed Strategy Profile Frequencies
    # ==================================================
    # Pivot df_tidy to have one row per replication round with each agent's strategy.
    profile_df = df_tidy.pivot_table(
        index=["simulation_id", "config_index", "replication_index", "round"],
        columns="agent",
        values="agent_strategy_id",
        aggfunc="first"
    ).reset_index()
    
    n_players = df_tidy.agent.nunique()

    # Rename pivoted columns for clarity. (Assumes agents are labelled 1 to n.)
    agent_col_renames = {i: f"agent{i}_strategy_id" for i in range(1, n_players+1)}
    profile_df = profile_df.rename(columns=agent_col_renames)

    # Create a combined strategy profile string (e.g., "cooperate-defect-retaliate").
    agent_cols = [f"agent{i}_strategy_id" for i in range(1, n_players+1)]
    profile_df["strategy_profile"] = ""
    for col in agent_cols[::-1]:
        profile_df["strategy_profile"] += profile_df[col].astype(str)
        if col != agent_cols[0]:
            profile_df["strategy_profile"] +=  "-"

    # Group by simulation_id, config_index, round, and strategy_profile to count replications.
    profile_counts = profile_df.groupby(
        ["simulation_id", "config_index", "round", "strategy_profile"]
    ).agg(count=("replication_index", "count")).reset_index()

    # Determine the total number of replications for each simulation/config/round.
    total_reps = profile_df.groupby(
        ["simulation_id", "config_index", "round"]
    ).size().reset_index(name="replications")

    # Merge counts and compute frequency as count divided by replications.
    observed_profile_freq = profile_counts.merge(total_reps, on=["simulation_id", "config_index", "round"])
    observed_profile_freq["frequency"] = observed_profile_freq["count"] / observed_profile_freq["replications"]

    # Create a column for each strategy profile frequency by reshaping observed_profile_freq
    
    observed_data = observed_profile_freq.pivot_table(
        index=["simulation_id", "config_index", "round"],
        columns="strategy_profile",
        values="frequency",
        aggfunc="first"
    ).reset_index()

    observed_data = observed_data.fillna(0)
    # Rename frequency columns to end in _frequency
    observed_data.columns = [f"{col}_frequency" if col != "simulation_id" and col != "config_index" and col != "round"
                            else col for col in observed_data.columns]

    # Merge with params_df to get the parameter values for each simulation and config
    observed_data = pandas.merge(observed_data, params_df, on=["simulation_id", "config_index"])
    
    return observed_data, observed_profile_freq

def compute_strategy_frequencies(df, recurrent_states):
    """
    For each row in df, compute the frequency that each player chooses each
    strategy.
    
    df must have columns named like "<state>_frequency". One for each state in
    recurrent_states. Otherwise, this function will throw a KeyError.
    
    Adds columns to df with names like "P<player_index>_strat_<strat>_likelihood".
    
    Also returns the player strategies found in recurrent states for later use.
    """
    # Identify number of players from recurrent_states
    n_players = len(recurrent_states[0].split("-"))

    # Construct a dictionary that for each player and a given strategy holds
    # a list of the recurrent_states where that player employes that strategy
    strat_states_mapping = {}
    for player_index in range(n_players):
        player_strats = numpy.unique([state.split("-")[player_index]
                                      for state in recurrent_states])
        player_states = {strat: [state for state in recurrent_states
                                 if state.split("-")[player_index] == strat]
                         for strat in player_strats}
        strat_states_mapping[f"P{player_index+1}"] = player_states

    # For each player, sum over the frequency columns corresponding to the strat
    for player in range(1, n_players+1):
        for strat, states in strat_states_mapping[f"P{player}"].items():
            cols = [f"{state}_frequency" for state in states]
            df[f"P{player}_strat_{strat}_frequency"] = df[cols].sum(axis=1)
    
    strat_states = [f"{player}_strat_{strat}"
                    for player, v in strat_states_mapping.items()
                    for v in v.keys()]
    
    return df, strat_states

# TODO: Refactor so that a weight_id_mapping_fn is unneccesary.
def get_sim_results(params, configs_df, params_df, weight_id_mapping_fn):

    sim_results = []
    for d in params:
        for k,v in d.items():
            if isinstance(v, list):
                d[k] = numpy.array(v)
        results = utils.thread_macro(d,
                            model_utils.create_profiles,
                            model_utils.apply_profile_filters,
                            payoffs.build_payoffs,
                            methods_egt.build_transition_matrix,
                            methods_egt.find_ergodic_distribution,
                            )
        
        weight_id_mapping = weight_id_mapping_fn(results)

        # Convert nested payoffs to a dataframe.
        data = {}
        # Make sure to filter out any payoffs which aren't relevant so that the
        # weight_id_mapping works as intended
        # Filter the result_payoffs keys for only those relevant to the sector_strategies
        sector_strategies = results["sector_strategies"]
        result_payoffs = results["payoffs"]
        result_payoffs = {k: v for k, v in result_payoffs.items()
                        if all([s in sector_strategies[f"S{i+1}"]
                                for i, s in enumerate(utils.string_to_tuple(k))])}
        for combination, v in result_payoffs.items():
            for player, payoff in v.items():
                weight_id = weight_id_mapping[combination][player]
                weight_name = f"payoffMatrix_weights_weight_{weight_id}_compare"
                data[weight_name] = payoff
        for i, v in enumerate(results["ergodic"].T):
            data[f"combination_{i}_frequency"] = v
        df = pandas.DataFrame(data)
        df["simulation_id"] = d["simulation_id"]
        df["config_index"] = d["config_index"]
        sim_results.append(df)

    sim_df = pandas.concat(sim_results).reset_index(drop=True)

    sim_df = sim_df.merge(params_df, on=["simulation_id", "config_index"])
    sim_df = sim_df.merge(configs_df, on=["simulation_id", "config_index"])
    
    return sim_df

# Load fairgame data
data_dir = "data"
plots_dir = "plots"
# 3 population models configs simulation ids
sim1 = "bellyfuls_skewering_expels_cc4dc882"
sim2 = "whiten_uncritical_chows_dc41924b"
# 4 population models configs simulation ids
sim3 = "fillers_preliminary_preppier_eba9e339"
sim4 = "welcomed_benefice_Kahlua_27c6ec76"

# TODO: Make sure to systematically go through all relevant values represented
# as data below. It would be possible to go through all the available directories
# and store them in one go if we wrap everything in a for loop. be careful because
# the sim ids are closely tied to which of the higher level directories we use
# as the external_data_dir

# Edit the following:
sims = [sim1]
external_data_dir = "external_data/fairgame_data/one-shot-results"
external_data_dir = "external_data/fairgame_data/Fairgame_results/OpenAIGPT4o/V1/one_shot"
# Whether to create a sim_df and run the second consistency check.
# Must be false for the 4 population model.
create_sim_df = True
# The following values are just used for naming plot files and not for processing
# the data
change_personality_for = "developer"
model_name = "3pop_full_trust"
game_type = "one_shot_game"
llm = "gpt4o"

# Take care to specify the simulation we are analysing!
# Currently, we only have one simulation per directory of fairgame results
sim_main = sims[0]
# Note: All of the filenames for the fairgame results we are analyzing contain
# a v1 or v2 to refer to whether the Users use a trust or conditional trust
# strategy. This is not a system I want to use long term and it would have
# been much better to incude the sim_id itself in the filename or better yet
# within the results file somehow. But for now, we have to specify the mapping
# of both v1 and v2 to the sim_id we are analyzing. For now, this is always
# sim_main.
filename_sim_mappings = {"v1": sim_main, "v2": sim_main}

# Fairgame results confusingly only list the labels and not the strategy ids for
# what each player chooses. Even if they did use the strategy ids known to
# Fairgame, this would not be consistent with the strategy_ids used in this
# repo. Here is a model specific mapping from the labels to the strategy ids.
# Note: Sometimes, Option C is not used. I even plan to and have started to
# switch to using only options A and B and strategy ids 5 and 6 for users only
# when running the 4 population model
strategy_id_mapping = {"regulator": {"Option A": 1, "Option B": 2, "Option C": 2},
                    "developer": {"Option A": 3, "Option B": 4, "Option C": 4},
                    "user": {"Option A": 5, "Option B": 6, "Option C": 7}}

# Unfortunately, we need to specify the state labels and recurrent states
# because I didn't consistently add them correctly to the params.json files.
# Note: In future, enforcing correct creation of params.json files is better

if model_name.startswith("3pop"):
    # for 3 populations, we might switch between models that use different
    # state labels and recurrent_states
    if model_name.contains("full_trust"):
        state_labels = ["T-C-C", "T-C-D", "T-D-C", "T-D-D",
                        "N-C-C", "N-C-D", "N-D-C", "N-D-D"]
        recurrent_states = ['5-3-1', '5-3-2', '5-4-1', '5-4-2', '6-3-1', '6-3-2', '6-4-1', '6-4-2']
    if model_name.contains("conditional_trust"):
        state_labels = ["CT-C-C", "CT-C-D", "CT-D-C", "CT-D-D",
                        "N-C-C", "N-C-D", "N-D-C", "N-D-D",]
        recurrent_states = ['7-3-1', '7-3-2', '7-4-1', '7-4-2', '6-3-1', '6-3-2', '6-4-1', '6-4-2']
if model_name.startswith("4pop"):
    # for 4 populations, we only need one set of labels:
    strategy_set=["C-T-C-C", "C-T-C-D", "C-T-D-C", "C-T-D-D",
                    "C-N-C-C", "C-N-C-D", "C-N-D-C", "C-N-D-D",
                    "D-CT-C-C", "D-CT-C-D", "D-CT-D-C", "D-CT-D-D",
                    "D-N-C-C", "D-N-C-D", "D-N-D-C", "D-N-D-D"]
    # Note: 7 is skipped to avoid confusion with the 3 population model
    recurrent_states = ['8-5-3-1',
     '8-5-3-2',
     '8-5-4-1',
     '8-5-4-2',
     '8-6-3-1',
     '8-6-3-2',
     '8-6-4-1',
     '8-6-4-2',
     '9-5-3-1',
     '9-5-3-2',
     '9-5-4-1',
     '9-5-4-2',
     '9-6-3-1',
     '9-6-3-2',
     '9-6-4-1',
     '9-6-4-2',]
strategy_state_mapping = dict(zip(state_labels, recurrent_states))

# ============================================
# Load and analyse the data as specified above

fairgame_data = load_fairgame_data({
    "data_dir": data_dir,
    "external_data_dir": external_data_dir,
    "sims": sims
})

params = fairgame_data["params"]
params_df = fairgame_data["params_df"]
configs = fairgame_data["configs"]
configs_df = fairgame_data["configs_df"]
results = fairgame_data["results"]
results_df = fairgame_data["results_df"]

# Log whether the state labels and recurrent states specified in this file
# are consistent with those saved to params.json files.

# Note: In future, we will hopefully rename strategy_set to state_labels
# everywhere so that future simulation runs store that as the keyword in
# params.json files.
for p in params:
    if "strategy_set" not in p:
        continue
    if p["strategy_set"] != state_labels:
        print("State labels in params.json file do not match those specified in this file. Ignore if this is an intentional workaround on your part.")
        print("state_labels in params.json file:", p["strategy_set"])
        print("state_labels in this file:", state_labels)
    if "recurrent_states" not in p:
        continue
    if p["recurrent_states"] != recurrent_states:
        print("Recurrent states in params.json file do not match those specified in this file. Ignore if this is an intentional workaround on your part.")
        print("recurrent_states in params.json file:", p["recurrent_states"])
        print("recurrent_states in this file:", recurrent_states)

df_wide = results_df
df_tidy = results_to_tidy_dataframe(df_wide)
df_tidy = add_indices_to_df(df_tidy, filename_sim_mappings)

# We also need to check that the payoff matrix computed from the params.json file
# parameters gives the correct payoff weights. We then need to check that those
# parameters plus payoffs give the same observations under our model as we reported
# in the paper

# Note: Data processing code here assumes that all models have the same number
# of strategies

def build_weight_ids_model1(results):
    """Build weight_id mapping for model 1 from the original paper."""
    # Build weight_id mapping
    weight_id_mapping = {}
    i=1
    # combinations are assumed to be inserted in order so both commands
    # should give the same vector of combinations
    combinations = list(results["payoffs"].keys())
    # combinations = numpy.sort(list(results["payoffs"].keys()))
    # Note: unfortunately, we have to assign different combinations to the same
    # weight_id because of how Fairgame works. This makes this code incredibly
    # brittle due to hardcoding. It will not work well if the combinations change.
    for combination in combinations:
        weight_id_mapping[combination] = {}
        for player in ["P1", "P2", "P3"]:
            weight_id = i
            weight_id_mapping[combination][player] = weight_id
            i+=1
    for player in ["P1", "P2", "P3"]:
        weight_id_mapping["7-3-1"][player] = weight_id_mapping["6-3-1"][player]
        weight_id_mapping["7-3-2"][player] = weight_id_mapping["6-3-2"][player]
        weight_id_mapping["7-4-1"][player] = weight_id_mapping["6-4-1"][player]
        weight_id_mapping["7-4-2"][player] = weight_id_mapping["6-4-2"][player]
    return weight_id_mapping

# TODO: Implement the following function for the 4 population models
# Only needed if we want to run the second consistency check
def build_weight_ids_model_four_pop(results):
    return None

consistency_check1(df_tidy)

if create_sim_df:
    sim_df = get_sim_results(params, configs_df, params_df, build_weight_ids_model1)
    consistency_check2(configs_df, sim_df)

# ==================================================
# Plots

def plot_strategy_distributions(df,
                                state_labels,
                                strategy_state_mapping,
                                filename_stub=""):
    """Plot the distribution of strategies of the given df for a harcoded set of parameters."""
    
    plot1 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 0.5) & (df["Eps"] == -0.1)],
                            state_labels,
                            x="b_fo",
                            x_label="b_fo",
                            title="Eps = -0.1",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    plot2 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 0.5) & (df["Eps"] == 0.2)],
                            state_labels,
                            x="b_fo",
                            x_label="b_fo",
                            title="Eps = 0.2",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    plot3 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 5) & (df["Eps"] == -0.1)],
                            state_labels,
                            x="b_fo",
                            x_label="b_fo",
                            title="Eps = -0.1",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    plot4 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 5) & (df["Eps"] == 0.2)],
                            state_labels,
                            x="b_fo",
                            x_label="b_fo",
                            title="Eps = 0.2",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    
    filename_start = f"llm_replication_{filename_stub}_llm_{llm}_{game_type}_personalities_{change_personality_for}_model_{model_name}"
    plots = {f"{filename_start}_cr_{0.5}_eps_{-0.1}": plot1,
             f"{filename_start}_cr_{0.5}_eps_{0.2}": plot2,
             f"{filename_start}_cr_{5}_eps_{-0.1}": plot3,
             f"{filename_start}_cr_{5}_eps_{0.2}": plot4}
    
    return plots


def plot_time_series_strategies(df,
                                state_labels,
                                strategy_state_mapping,
                                filename_stub=""):
    """Plot a time series of strategies of the given df for a harcoded set of parameters."""
    
    plot1 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 0.5) & (df["Eps"] == -0.1)],
                            state_labels,
                            x="round",
                            x_label="Round",
                            title="Eps = -0.1",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    plot2 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 0.5) & (df["Eps"] == 0.2)],
                            state_labels,
                            x="round",
                            x_label="Round",
                            title="Eps = 0.2",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    plot3 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 5) & (df["Eps"] == -0.1)],
                            state_labels,
                            x="round",
                            x_label="Round",
                            title="Eps = -0.1",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    plot4 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 5) & (df["Eps"] == 0.2)],
                            state_labels,
                            x="round",
                            x_label="Round",
                            title="Eps = 0.2",
                            thresholds=None,
                            stacked=False,
                            strategy_state_mapping=strategy_state_mapping,
                            )
    
    filename_start = f"llm_replication_{filename_stub}_llm_{llm}_{game_type}_personalities_{change_personality_for}_model_{model_name}"
    plots = {f"{filename_start}_cr_{0.5}_eps_{-0.1}": plot1,
             f"{filename_start}_cr_{0.5}_eps_{0.2}": plot2,
             f"{filename_start}_cr_{5}_eps_{-0.1}": plot3,
             f"{filename_start}_cr_{5}_eps_{0.2}": plot4}
    
    return plots

observed_data, observed_profile_freq = df_to_observed_data(df_tidy, params_df, strategy_id_mapping)
for state in recurrent_states:
    if f"{state}_frequency" not in observed_data.columns:
        observed_data[f"{state}_frequency"] = 0
observed_data, states_labels_compact = compute_strategy_frequencies(observed_data, recurrent_states)

# Ensure that only one set of simulation results is plotted at a time!
df = observed_data[observed_data["simulation_id"] == sim_main]

if game_type == "one_shot_game":
    
    plots = plot_strategy_distributions(df, state_labels, strategy_state_mapping, filename_stub="one_shot")
    plot_save_id = data_utils.create_id()
    data_utils.save_plots(plots, plots_dir=f"plots/fairgame_replication_plots/one_shot_games/{plot_save_id}")

if game_type == "repeated_game":
    plots = {}
    
    # df (the observed_data) has columns for the frequency of each strategy profile
    # (also called recurrent states) and columns for the frequency of each
    # player's strategy (across recurrent states). We can specify which set of
    # columns to plot by specifying either state_labels (which contains the labels of the
    # recurrent states; yes, it's a bit of a misnomer) or states_labels_compact (for the player strategies).
    # TODO: perhaps relable state_labels to recurrent_state_labels to avoid
    # confusion in future.
    
    # First plot the final round frequencies for each state
    final_round = df["round"].max()
    final_round_df = final_round[final_round["round"] == final_round]
    # Note: assumes all games last the same number of rounds
    plots = {**plots, **plot_strategy_distributions(df, state_labels, strategy_state_mapping, strategy_state_mapping)}

    # We then want to plot the average frequences across rounds

    df_avg = df.groupby(["simulation_id", "config_index"]).mean()
    plots = {**plots, **plot_strategy_distributions(df_avg, state_labels, strategy_state_mapping, filename_stub="average_round")}
    
    # Plot each player's final and average strategy frequencies across states
    plots = {**plots, **plot_strategy_distributions(final_round_df, states_labels_compact, filename_stub="player_strategy_frequencies_final")}
    plots = {**plots, **plot_strategy_distributions(df_avg, states_labels_compact, filename_stub="player_strategy_frequencies_average")}
    
    # Plot the strategy frequencies per round
    for round in range(final_round + 1):
        round_df = df[df["round"] == round]
        plots = {**plots, **plot_strategy_distributions(round_df, state_labels, strategy_state_mapping, filename_stub=f"round_{round}")}
        plots = {**plots, **plot_strategy_distributions(round_df, states_labels_compact, filename_stub=f"player_strategy_frequencies_round_{round}")}
    
    # Plot a time series for each value of the config id
    for config_index in df["config_index"].unique():
        config_df = df[df["config_index"] == config_index]
        # Only b_fo changes when config_index changes, so that's all we add to the filename
        b_fo = config_df["b_fo"].unique()[0]
        plots = {**plots, **plot_time_series_strategies(config_df, state_labels, strategy_state_mapping, filename_stub=f"time_series_b_fo_{b_fo}")}
        plots = {**plots, **plot_time_series_strategies(config_df, states_labels_compact, filename_stub=f"time_series_b_fo_{b_fo}")}
    
    plot_save_id = data_utils.create_id()
    data_utils.save_plots(plots, plots_dir=f"plots/fairgame_replication_plots/repeated_games/{plot_save_id}")
