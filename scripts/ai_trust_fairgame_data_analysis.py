import logging
import os
import sys

import prototype_phd.data_utils as data_utils
import prototype_phd.methods.egt as methods_egt
import prototype_phd.models as models
import prototype_phd.model_utils as model_utils
import prototype_phd.payoffs as payoffs
import prototype_phd.plot_utils as plot_utils
import prototype_phd.utils as utils

import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
    # Some columns of df have lists as values but they are currently represented as strings.
    # We need to convert them to lists
    for i in range(1, n_players+1):
        for col_name in [f"agent{i}_scores", f"agent{i}_strategies", f"agent{i}_messages"]:
            if isinstance(df[col_name].iloc[0], str):
                df[col_name] = df[col_name].apply(lambda x: eval(x))
    
    # Verify that the dataframe stores round lists consistently
    # All round-based columns must share the same length equal to rounds_played
    # n_rounds = results_df["played_rounds"].unique()[0]
    # for i in range(1, n_players+1):
    #     for col_name in [f"agent{i}_scores", f"agent{i}_strategies", f"agent{i}_messages"]:
    #         # logging.info(f"Checking {col_name}")
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
    
    assert parts[-2] == "en"
    if version not in ["v1", "v2"]:
        # Filename format has changed. Assume the version is v1 as it likely
        # isn't used anymore.
        version = "v1"
    return version, config_index, replication_index

def add_indices_to_df(df, sim_mapping):
    """Add indices to results df.
    
    We can use the filename column to identify the params.json file for each game
    Each params_df row has a simulation_id and index. We know the mapping from
    v1, v2 to the relevant simulation id.
    The filename column in df_tidy contains both the version number and the index
    number, as well as the replication number.
    """
    # print("Adding indices to dataframe.")
    # print("Extracting version and index from filename.")
    # logging.info(f"sim_mapping: {sim_mapping}")
    # logging.info(f"filenames: { df['filename'].unique() }")

    df["simulation_id"] = df['filename'].apply(lambda x: sim_mapping[extract_version_index(x)[0]])
    df["config_index"] = df['filename'].apply(lambda x: extract_version_index(x)[1])
    df["replication_index"] = df['filename'].apply(lambda x: extract_version_index(x)[2])
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
            logging.info(f"Game {game}, Round {rnd}, {agent_str}: {msg}")
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
            logging.info(f"Column {col} has {n_mismatches} mismatches.")
            logging.info(f"Expected values: {x1}")
            logging.info(f"Actual values: {x2}")
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
        player_strats = numpy.unique([state.split("-")[::-1][player_index]
                                      for state in recurrent_states])
        player_states = {strat: [state for state in recurrent_states
                                 if state.split("-")[::-1][player_index] == strat]
                         for strat in player_strats}
        strat_states_mapping[f"P{player_index+1}"] = player_states

    # For each player, sum over the frequency columns corresponding to the strat
    for player in range(1, n_players+1):
        for strat, states in strat_states_mapping[f"P{player}"].items():
            cols = [f"{state}_frequency" for state in states]
            df[f"P{player}_strat_{strat}_frequency"] = df[cols].sum(axis=1)
    
    strat_states = [f"{player}_strat_{strat}"
                    for player, v in strat_states_mapping.items()
                    for strat in v.keys()]
    
    return df, strat_states

def compact_strategy_labels(strategy_labels):
    """
    Given a list of strategy labels in the form "P{player}_strat_{strat}",
    return a new list where for each player the last strategy (in order of appearance)
    is dropped.
    """
    per_player = {}
    # Group labels by player
    for label in strategy_labels:
        player = label.split("_")[0]  # e.g., "P1"
        per_player.setdefault(player, []).append(label)
    
    compact = []
    for player, labels in per_player.items():
        # Assume the order of labels is the order in which they were generated.
        # Drop the last strategy if there is more than one for this player.
        if len(labels) > 1:
            compact.extend(labels[:-1])
        else:
            compact.extend(labels)
    return compact

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

def run_data_analysis(args):
    names = ["model_name", "game_type", "llm", "change_personality_for"]
    names2 = ["data_dir",
              "plots_dir",
              "external_data_dir",
              "sims"]
    model_name, game_type, llm, change_personality_for = [args[k] for k in names]
    data_dir, plots_dir, external_data_dir, sims = [args[k] for k in names2]

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
                        "user": {"Option A": 5, "Option B": 6, "Option C": 7},
                        "commentariat": {"Option A": 8, "Option B": 9},}

    # Unfortunately, we need to specify the state labels and recurrent states
    # because I didn't consistently add them correctly to the params.json files.
    # Note: In future, enforcing correct creation of params.json files is better
    cmap = plt.colormaps["tab20"]
    if model_name.startswith("3pop"):
        # for 3 populations, we might switch between models that use different
        # state labels and recurrent_states
        # With only 8 recurrent states, we should use the following colobar to be
        # consistent with the original paper
        cmap = ListedColormap(["red", "brown", "orange", "lightblue", "pink", "green", "mediumblue", "black"])
        if "full_trust" in model_name:
            state_labels = ["T-C-C", "T-C-D", "T-D-C", "T-D-D",
                            "N-C-C", "N-C-D", "N-D-C", "N-D-D"]
            recurrent_states = ['5-3-1', '5-3-2', '5-4-1', '5-4-2', '6-3-1', '6-3-2', '6-4-1', '6-4-2']
        if "conditional_trust" in model_name:
            state_labels = ["CT-C-C", "CT-C-D", "CT-D-C", "CT-D-D",
                            "N-C-C", "N-C-D", "N-D-C", "N-D-D",]
            recurrent_states = ['7-3-1', '7-3-2', '7-4-1', '7-4-2', '6-3-1', '6-3-2', '6-4-1', '6-4-2']
        # Whether to create a sim_df and run the second consistency check.
        # Must be false for the 4 population model.
        create_sim_df = True
    if model_name.startswith("4pop"):
        # With 16 recurernt states, we need to use a different colormap
        cmap = plt.colormaps["tab20"]
        # for 4 populations, we only need one set of labels:
        state_labels=["C-T-C-C", "C-T-C-D", "C-T-D-C", "C-T-D-D",
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
        # Whether to create a sim_df and run the second consistency check.
        # Must be false for the 4 population model.
        create_sim_df = False
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
    
    if "4pop" in model_name:
        print("4 population model")
        print("params_df cols: ", params_df.columns)
        print("cI in params_df", "cI" in params_df.columns)
        print("cW in params_df", "cW" in params_df.columns)
        print("bI in params_df", "bI" in params_df.columns)
        # We need to rename the cl and bl columns to cI and bI
        
        # Rename the columns to match the expected names
        params_df = params_df.rename(columns={"cl": "cI", "bl": "bI"})
        # # Rename the keys in params too
        # for p in params:
        #     p["cI"] = p.pop("cl")
        #     p["bI"] = p.pop("bl")

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
    
    print("df_wide cols: ", df_wide.columns)
    print("df_wide_agent1_personality: ", df_wide["agent1_personality"].unique())
    print("df_wide_agent2_personality: ", df_wide["agent2_personality"].unique())
    print("df_wide_agent3_personality: ", df_wide["agent3_personality"].unique())
    df_tidy = results_to_tidy_dataframe(df_wide)
    df_tidy = add_indices_to_df(df_tidy, filename_sim_mappings)
    df_tidy = df_tidy.merge(params_df, on=["simulation_id", "config_index"])
    df_tidy = df_tidy.merge(configs_df, on=["simulation_id", "config_index"])
    df_tidy["game_id"] = df_tidy["simulation_id"] + "_" + df_tidy["config_index"].astype(str) + "_" + df_tidy["replication_index"].astype(str)

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
    
    if "3pop" in model_name:
        def plot_strategy_distributions(df,
                                        state_labels,
                                        strategy_state_mapping,
                                        x="b_fo",
                                        x_label="b_fo",
                                        title="Eps = -0.1",
                                        cmap=cmap,
                                        filename_stub=""):
            """Plot the distribution of strategies of the given df for a harcoded set of parameters."""
            
            filename_start = f"llm_replication_{filename_stub}_llm_{llm}_{game_type}_personalities_{change_personality_for}_model_{model_name}"
            plots = {}
            df1 = df[(df["cR"] == 0.5) & (df["Eps"] == -0.1)]
            df2 = df[(df["cR"] == 0.5) & (df["Eps"] == 0.2)]
            df3 = df[(df["cR"] == 5) & (df["Eps"] == -0.1)]
            df4 = df[(df["cR"] == 5) & (df["Eps"] == 0.2)]
            if len(df1) > 1:
                plot1 = plot_utils.plot_strategy_distribution(df1,
                                        state_labels,
                                        x=x,
                                        x_label=x_label,
                                        title="Eps = -0.1",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{0.5}_eps_{-0.1}": plot1,}
            if len(df2) > 1:
                plot2 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 0.5) & (df["Eps"] == 0.2)],
                                        state_labels,
                                        x=x,
                                        x_label=x_label,
                                        title="Eps = 0.2",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{0.5}_eps_{0.2}": plot2,}
            if len(df3) > 1:
                plot3 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 5) & (df["Eps"] == -0.1)],
                                        state_labels,
                                        x=x,
                                        x_label=x_label,
                                        title="Eps = -0.1",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{5}_eps_{-0.1}": plot3,}
            if len(df4) > 1:
                plot4 = plot_utils.plot_strategy_distribution(df[(df["cR"] == 5) & (df["Eps"] == 0.2)],
                                        state_labels,
                                        x=x,
                                        x_label=x_label,
                                        title="Eps = 0.2",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{5}_eps_{0.2}": plot4,}
            
            return plots


        def plot_time_series_strategies(df,
                                        state_labels,
                                        strategy_state_mapping,
                                        cmap=cmap,
                                        filename_stub=""):
            """Plot a time series of strategies of the given df for a harcoded set of parameters."""
            filename_start = f"llm_replication_{filename_stub}_llm_{llm}_{game_type}_personalities_{change_personality_for}_model_{model_name}"
            plots = {}
            df1 = df[(df["cR"] == 0.5) & (df["Eps"] == -0.1)]
            df2 = df[(df["cR"] == 0.5) & (df["Eps"] == 0.2)]
            df3 = df[(df["cR"] == 5) & (df["Eps"] == -0.1)]
            df4 = df[(df["cR"] == 5) & (df["Eps"] == 0.2)]
            
            if len(df1) > 1:
                plot1 = plot_utils.plot_strategy_distribution(df1,
                                        state_labels,
                                        x="round",
                                        x_label="Round",
                                        title="Eps = -0.1",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{0.5}_eps_{-0.1}": plot1,}
            if len(df2) > 1:
                plot2 = plot_utils.plot_strategy_distribution(df2,
                                        state_labels,
                                        x="round",
                                        x_label="Round",
                                        title="Eps = 0.2",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{0.5}_eps_{0.2}": plot2,}
            if len(df3) > 1:
                plot3 = plot_utils.plot_strategy_distribution(df3,
                                        state_labels,
                                        x="round",
                                        x_label="Round",
                                        title="Eps = -0.1",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{5}_eps_{-0.1}": plot3,}
            if len(df4) > 1:
                plot4 = plot_utils.plot_strategy_distribution(df4,
                                        state_labels,
                                        x="round",
                                        x_label="Round",
                                        title="Eps = 0.2",
                                        thresholds=None,
                                        stacked=False,
                                        strategy_state_mapping=strategy_state_mapping,
                                        cmap=cmap,
                                        )
                plots = {**plots, f"{filename_start}_cr_{5}_eps_{0.2}": plot4,}

            return plots
    
    if "4pop" in model_name:
        
        def plot_strategy_distributions(df,
                                        state_labels,
                                        strategy_state_mapping,
                                        x="bI",
                                        x_label="Benefit to investigators, bI",
                                        title="",
                                        cmap=cmap,
                                        filename_stub=""):
            """Plot the distribution of strategies of the given df for a harcoded set of parameters."""
            
            filename_start = f"llm_replication_{filename_stub}_llm_{llm}_{game_type}_personalities_{change_personality_for}_model_{model_name}"
            plots = {}
            
            # we need to improve the plotting here.
            # Use a line plot instead.
            
            for cW in [0, 5, 10]:
                for cI in [0.5, 5]:
                    df1 = df[(df["cI"] == cI) & (df["cW"] == cW)]
                    if len(df1) > 1:
                        plot_title = f"{title}cI={cI}, cW={cW}"
                        y_label='Frequency'
                        if strategy_state_mapping!=None:
                            recurrent_states = [strategy_state_mapping[strategy]
                                                for strategy in state_labels]
                        else:
                            recurrent_states = state_labels    
                        fig, ax = plt.subplots()
                        # Define a list of marker shapes to distinguish each strategy.
                        markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>', '1', '2', '3', '4', 'h', 'H', '+', 'x', '|', '_']
                        # Plot scatter points for each strategy with a unique marker.
                        for i, state in enumerate(recurrent_states):
                            marker = markers[i % len(markers)]
                            x_values = df1[x].values
                            jitter_size = 0.1
                            jitter = numpy.random.uniform(-jitter_size, jitter_size, size=x_values.shape)
                            jittered_x = x_values + jitter
                            y_values = df1[state + "_frequency"].values
                            # sort the jittered values so the line connects them
                            sort_idx = numpy.argsort(jittered_x)
                            sorted_x = jittered_x[sort_idx]
                            sorted_y = y_values[sort_idx]
                            # If there are only 4 states, then I want to use
                            # custom labels
                            if len(state_labels) == 4:
                                custom_labels = {"P1_strat_1": "Regulator Cooperates",
                                                 "P2_strat_3": "Developer Cooperates",
                                                 "P3_strat_5": "User Trusts",
                                                 "P4_strat_8": "Commentariat Cooperates"}
                                label = custom_labels[i]
                            else:
                                label = state
                            # Plot connected points with markers and a line between them
                            ax.plot(sorted_x,
                                    sorted_y,
                                    color=cmap(i),
                                    marker=marker,
                                    markersize=20,   # adjust as needed
                                    linestyle='-',
                                    linewidth=1,
                                    label=label)

                        # ax.legend(loc='upper left')                 
                        # Move legend outside the figure
                        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.legend(bbox_to_anchor=(1.05, 1),
                                loc='upper left',
                                markerscale=0.5,      # reduce marker size in legend
                                # handlelength=2,       # length of the legend handle
                                # handletextpad=0.5  # space between marker and text
                                )   
                        ax.set_title(plot_title)
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        plt.tight_layout()
            
                        plots = {**plots, f"{filename_start}_cI_{cI}_cW_{cW}": fig}
            
            return plots


        def plot_time_series_strategies(df,
                                        state_labels,
                                        strategy_state_mapping,
                                        title="",
                                        cmap=cmap,
                                        filename_stub=""):
            """Plot a time series of strategies of the given df for a harcoded set of parameters."""
            filename_start = f"llm_replication_{filename_stub}_llm_{llm}_{game_type}_personalities_{change_personality_for}_model_{model_name}"
            plots = {}

            for cW in [0, 5, 10]:
                for cI in [0.5, 5]:
                    df1 = df[(df["cI"] == cI) & (df["cW"] == cW)]
                    if len(df1) > 1:
                        plot1 = plot_utils.plot_strategy_distribution(df1,
                                                state_labels,
                                                x="round",
                                                x_label="Round",
                                                title=f"{title}_cI={cI}, cW={cW}",
                                                thresholds=None,
                                                stacked=False,
                                                strategy_state_mapping=strategy_state_mapping,
                                                cmap=cmap,
                                                )
                        plots = {**plots, f"{filename_start}_cI_{cI}_cW_{cW}": plot1,}

            return plots
    
    # TODO: We need to filter by agent personalities given change_personality_for (only one agent sees a change in personality at a time)
    # print("df: columns", df_tidy.columns)
    # print("df.agent_personality", df_tidy.agent_personality.unique())
    # raise ValueError("Stop here")

    observed_data, observed_profile_freq = df_to_observed_data(df_tidy, params_df, strategy_id_mapping)
    for state in recurrent_states:
        if f"{state}_frequency" not in observed_data.columns:
            observed_data[f"{state}_frequency"] = 0
    observed_data, states_labels_compact = compute_strategy_frequencies(observed_data, recurrent_states)
    # For the compact states labels, we want to exclude the last strategy for each player because
    # that information is redundant.

    # TODO: Improve states_labels_compact labelling to be easier to interpret
    states_labels_compact = compact_strategy_labels(states_labels_compact)
    state_mapping_compact = dict(zip(states_labels_compact, states_labels_compact))
    # Ensure that only one set of simulation results is plotted at a time!
    df = observed_data[observed_data["simulation_id"] == sim_main]

    if game_type == "one_shot_game":
        
        plots = plot_strategy_distributions(df, state_labels, strategy_state_mapping, filename_stub="recurrent_state_frequencies")
        # Plot each player's strategy frequencies across states
        plots = {**plots, **plot_strategy_distributions(df, states_labels_compact, state_mapping_compact, filename_stub="player_strategy_frequencies")}

        data_utils.save_plots(plots, plots_dir=plots_dir)

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
        final_round_df = df[df["round"] == final_round]
        # Note: assumes all games last the same number of rounds
        plots = {**plots, **plot_strategy_distributions(final_round_df, state_labels, strategy_state_mapping, filename_stub="final_round")}

        # We then want to plot the average frequences across rounds
        
        def safe_mean_agg(x):
            if pandas.api.types.is_numeric_dtype(x):
                return x.mean()
            else:
                return x.iloc[0]

        df_avg = df.groupby(["simulation_id", "config_index"]).agg(safe_mean_agg)
        plots = {**plots, **plot_strategy_distributions(df_avg, state_labels, strategy_state_mapping, filename_stub="average_round")}
        
        # Plot each player's final and average strategy frequencies across states
        plots = {**plots, **plot_strategy_distributions(final_round_df, states_labels_compact, state_mapping_compact, filename_stub="player_strategy_frequencies_final")}
        plots = {**plots, **plot_strategy_distributions(df_avg, states_labels_compact, state_mapping_compact, filename_stub="player_strategy_frequencies_average")}
        
        # Plot the strategy frequencies per round
        for round in range(final_round + 1):
            round_df = df[df["round"] == round]
            plots = {**plots, **plot_strategy_distributions(round_df, state_labels, strategy_state_mapping, filename_stub=f"round_{round}")}
            plots = {**plots, **plot_strategy_distributions(round_df, states_labels_compact, state_mapping_compact, filename_stub=f"player_strategy_frequencies_round_{round}")}
        
        # Plot a time series for each value of the config id
        for config_index in df["config_index"].unique():
            config_df = df[df["config_index"] == config_index]
            # Only b_fo changes when config_index changes, so that's all we add to the filename
            b_fo = config_df["b_fo"].unique()[0]
            plots = {**plots, **plot_time_series_strategies(config_df, state_labels, strategy_state_mapping, filename_stub=f"time_series_b_fo_{b_fo}")}
            plots = {**plots, **plot_time_series_strategies(config_df, states_labels_compact, state_mapping_compact, filename_stub=f"time_series_b_fo_{b_fo}")}

        data_utils.save_plots(plots, plots_dir=plots_dir)
        
    plt.close('all')

# Load fairgame data
data_dir = "data"
plots_dir = "plots"
plot_save_id = data_utils.create_id()
data_utils.setup_logging(log_path=f"logs/fairgame_analysis/{plot_save_id}.log")
# 3 population models configs simulation ids
sim1 = "bellyfuls_skewering_expels_cc4dc882"
sim2 = "whiten_uncritical_chows_dc41924b"
# 4 population models configs simulation ids
sim3 = "fillers_preliminary_preppier_eba9e339"
sim4 = "welcomed_benefice_Kahlua_27c6ec76"

folder_to_sim_mapping = {
    # We longer use the one-shot-results folder: this was just for practise
    # "one-shot-results": sim1,
    # The baseline results for both 3 population models
    "Fairgame_results/OpenAIGPT4o/V1/one_shot": sim1,
    "Fairgame_results/OpenAIGPT4o/V2/one_shot": sim2,
    "Fairgame_results/MistralLarge/V1/one_shot": sim1,
    "Fairgame_results/MistralLarge/V2/one_shot": sim2,
    "Fairgame_results/OpenAIGPT4o/V1/repeated": sim1,
    "Fairgame_results/OpenAIGPT4o/V2/repeated": sim2,
    "Fairgame_results/MistralLarge/V1/repeated": sim1,
    "Fairgame_results/MistralLarge/V2/repeated": sim2,
    # Personality simulations only used the conditional trust 3 population model
    # and are only for one-shot games
    "3p_with_perso/OpenAIGPT4o/20250311_developer_fix": sim2,
    "3p_with_perso/OpenAIGPT4o/20250311_regulator_fix": sim2,
    "3p_with_perso/OpenAIGPT4o/20250311_user_fix": sim2,
    "3p_with_perso/MistralLarge/20250311_developer_fix": sim2,
    "3p_with_perso/MistralLarge/20250311_regulator_fix": sim2,
    "3p_with_perso/MistralLarge/20250311_user_fix": sim2,
    # The 4 population models
    # v1 is for the model where commentariat investigates developers
    # v2 if for the model where commentariat investigate the regulators
    # Again, these rsults are only for one-shot games.
    "results_4p_v1/OpenAIGPT4o": sim3,
    "results_4p_v1/MistralLarge": sim3,
    "4p_v2_results/OpenAIGPT4o": sim4,
    "4p_v2_results/MistralLarge": sim4,
}

input_to_dir_mapping = {
    "3pop_full_trust_one_shot_game_gpt4o_personality_none": "Fairgame_results/OpenAIGPT4o/V1/one_shot",
    "3pop_conditional_trust_one_shot_game_gpt4o_personality_none": "Fairgame_results/OpenAIGPT4o/V2/one_shot",
    "3pop_full_trust_repeated_game_gpt4o_personality_none": "Fairgame_results/OpenAIGPT4o/V1/repeated",
    "3pop_conditional_trust_repeated_game_gpt4o_personality_none": "Fairgame_results/OpenAIGPT4o/V2/repeated",
    "3pop_full_trust_one_shot_game_mistral_large_personality_none": "Fairgame_results/MistralLarge/V1/one_shot",
    "3pop_conditional_trust_one_shot_game_mistral_large_personality_none": "Fairgame_results/MistralLarge/V2/one_shot",
    "3pop_full_trust_repeated_game_mistral_large_personality_none": "Fairgame_results/MistralLarge/V1/repeated",
    "3pop_conditional_trust_repeated_game_mistral_large_personality_none": "Fairgame_results/MistralLarge/V2/repeated",
    "3pop_conditional_trust_one_shot_game_gpt4o_personality_developer": "3p_with_perso/OpenAIGPT4o/20250311_developer_fix",
    "3pop_conditional_trust_one_shot_game_gpt4o_personality_regulator": "3p_with_perso/OpenAIGPT4o/20250311_regulator_fix",
    "3pop_conditional_trust_one_shot_game_gpt4o_personality_user": "3p_with_perso/OpenAIGPT4o/20250311_user_fix",
    "3pop_conditional_trust_one_shot_game_mistral_large_personality_developer": "3p_with_perso/MistralLarge/20250311_developer_fix",
    "3pop_conditional_trust_one_shot_game_mistral_large_personality_regulator": "3p_with_perso/MistralLarge/20250311_regulator_fix",
    "3pop_conditional_trust_one_shot_game_mistral_large_personality_user": "3p_with_perso/MistralLarge/20250311_user_fix",
    "4pop_v1_one_shot_game_gpt4o_personality_none": "results_4p_v1/OpenAIGPT4o",
    "4pop_v1_one_shot_game_mistral_large_personality_none": "results_4p_v1/MistralLarge",
    "4pop_v2_one_shot_game_gpt4o_personality_none": "4p_v2_results/OpenAIGPT4o",
    "4pop_v2_one_shot_game_mistral_large_personality_none": "4p_v2_results/MistralLarge",
}

# Edit the following:
# All possible values:
set_change_personality_for = ["none", "developer", "regulator", "user"]
set_model_name = ["3pop_full_trust", "3pop_conditional_trust", "4pop_v1", "4pop_v2"]
set_game_type = ["one_shot_game", "repeated_game"]
set_llm = ["gpt4o", "mistral_large"]
# Constraints:
# We only have results for the 3 population model with personality for the
# conditional trust model and only for one-shot games.
# We only have results for the 4 population model for one-shot games.

for model_name in set_model_name:
    for change_personality_for in set_change_personality_for:
        for game_type in set_game_type:
            for llm in set_llm:
                if (change_personality_for != "none") and ((model_name != "3pop_conditional_trust") or (game_type != "one_shot_game")):
                    continue
                if (game_type != "one_shot_game") and ("4pop" in model_name):
                    continue
                selection = f"{model_name}_{game_type}_{llm}_personality_{change_personality_for}"
                external_data_dir_stub = "external_data/fairgame_data"
                assert selection in input_to_dir_mapping.keys()
                external_data_dir_suffix = input_to_dir_mapping[selection]
                external_data_dir = f"{external_data_dir_stub}/{external_data_dir_suffix}"
                assert os.path.exists(external_data_dir)
                assert external_data_dir_suffix in folder_to_sim_mapping.keys()
                sims = [folder_to_sim_mapping[external_data_dir_suffix]]
                plots_dir_full = f"{plots_dir}/fairgame_replication_plots/{plot_save_id}/{model_name}/{game_type}/{llm}/personalities_{change_personality_for}"
                run_data_analysis({
                    "data_dir": data_dir,
                    "plots_dir": plots_dir_full,
                    "external_data_dir": external_data_dir,
                    "sims": sims,
                    "model_name": model_name,
                    "game_type": game_type,
                    "llm": llm,
                    "change_personality_for": change_personality_for})
