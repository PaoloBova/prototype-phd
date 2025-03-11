import logging
import numpy
import prototype_phd.data_utils as data_utils
import prototype_phd.models as models
import prototype_phd.model_utils as model_utils
import prototype_phd.payoffs as payoffs
import prototype_phd.utils as utils

def build_fairgame_configs(data: dict) -> dict:
    """
    Expects 'data' to include:
      - "result_payoffs": dict mapping combination -> player -> payoffs
      - "strategy_dict": dict for strategies (e.g., {"en": {"strategy1": "Defect", ...}})
      - "strategy_mapping": dict mapping number-string to strategy key (e.g., {"1": "strategy1", ...})
      - "fairgame_initial_config": dict with fairgame configuration without payoffMatrix
    Returns a list of fairgame configurations, one per payoff matrix.
    """

    # The payoffMatrix in a FAIRGAME matrix should be in the following form:

    # {"strategies": {"strategy1": "Cooperate", "strategy2": "Defect"},
    #  "weights": {"weight1": 0.5, "weight2": 0.5},
    #  "combinations": {"combination1": [["strategy1", "weight1"]
    #                                    ["strategy2", "weight2"]]}}

    # Our payoffs in results are in the form combination->player->payoff
    # {"1-1": {"P1": np.array, "P2": np.array}}
    # We need to create a new payoff matrix for each row in our arrays.
    # To convert this given a row of each array, we need to create a new weight
    # key for each value and a new strategy key for each strategy in the combination.

    result_payoffs = data["result_payoffs"]
    strategy_dict = data["strategy_dict"]
    strategy_mapping = data["strategy_mapping"]
    initial_config = data["fairgame_initial_config"]
    
    # Filter the result_payoffs keys for only those relevant to the strategy_dict
    # and strategy_mapping
    result_payoffs = {k: v for k, v in result_payoffs.items()
                      if all([strategy_mapping[str(s)] in strategy_dict["en"].keys()
                              for s in utils.string_to_tuple(k)])}
    
    # Determine the number of payoff matrices (n_matrices)
    n_matrices = 1
    for v1 in result_payoffs.values():
        for v2 in v1.values():
            if isinstance(v2, numpy.ndarray):
                # We need to do this because some values may be the 0 integer
                # instead of a numpy array
                n_matrices = max(n_matrices, len(v2))

    # Compute combinations_dict (structure remains same for each config;
    # notice that we do not use the array values of result_payoffs)
    combinations_dict = {}
    i = 0
    j = 0
    print(f"result_payoffs {result_payoffs.keys()}")
    for k1, v1 in result_payoffs.items():
        i += 1
        # Reverse the keys in the dictionary to get the correct order
        v1_keys = list(v1.keys())
        v1_values = list(v1.values())
        v1_reversed = dict(zip(v1_keys[::-1], v1_values[::-1]))
        combinations_dict[f"combination{i}"] = []
        strategies = utils.string_to_tuple(k1)
        for s in strategies:
            j += 1
            combination_val = [strategy_mapping[f"{s}"], f"weight_{j}"]
            combinations_dict[f"combination{i}"].append(combination_val)
    
    fairgame_configs = []
    # Create a fairgame_config per payoff matrix index
    for idx in range(n_matrices):
        weights_dict = {}
        i = 0
        for k1, v1 in result_payoffs.items():
            # Reverse the keys in the dictionary to get the correct order
            v1_keys = list(v1.keys())
            v1_values = list(v1.values())
            v1_reversed = dict(zip(v1_keys[::-1], v1_values[::-1]))
            for k2, v2 in v1_reversed.items():
                i += 1
                # If v2 is a numpy array, select the value at index 'idx'
                if isinstance(v2, numpy.ndarray):
                    if len(v2) > idx:
                        weight_val = round(float(v2[idx]), 1)
                    else:
                        logging.error(f"Index {idx} out of bounds for combination {k1}->{k2}")
                        raise IndexError
                else:
                    weight_val = round(float(v2), 1)  # integer or float case
                weights_dict[f"weight_{i}"] = weight_val

            payoff_matrix = {
            "strategies": strategy_dict,
            "weights": weights_dict,
            "combinations": combinations_dict,
        }
        fairgame_config = {**initial_config, "payoffMatrix": payoff_matrix}
        fairgame_configs.append(fairgame_config)
    
    return fairgame_configs

simulation_id, current_commit, data_dir, plots_dir = data_utils.setup_project()

params = {**models.build_ai_trust(Eps=[-0.1],
                                  cR=[5],
                                  b_fo=[5],
                                  cW=[0, 5, 10],
                                  pW=[0.5],
                                  bl=[0, 5, 10],
                                  cl=[0.5, 5],
                           ),
          "strategy_set": ["C-T-C-C", "C-T-C-D", "C-T-D-C", "C-T-D-D",
                    "C-N-C-C", "C-N-C-D", "C-N-D-C", "C-N-D-D",
                    "D-CT-C-C", "D-CT-C-D", "D-CT-D-C", "D-CT-D-D",
                    "D-N-C-C", "D-N-C-D", "D-N-D-C", "D-N-D-D"],
          "simulation_id": simulation_id,
          "commit": current_commit,
          "dispatch-type": 'multiple-populations',
          "payoffs_key": "ai-trust-v2",
          "Z": {"S4": 100, "S3": 100, "S2": 100, "S1": 100},
          "allowed_sectors": {"P4": ["S4"],
                              "P3": ["S3"],
                              "P2": ["S2"],
                              "P1": ["S1"], },
          "sector_strategies": {"S4": [8, 9],
                                "S3": [6, 7],
                                "S2": [3, 4],
                                "S1": [1, 2], },
          }

results = utils.thread_macro(params,
                       model_utils.create_profiles,
                       model_utils.apply_profile_filters,
                       payoffs.build_payoffs,
                       )

strategy_dict = {"en": {"strategy1": "Option A",
                        "strategy2": "Option B",
                        }}
strategy_mapping = {"1": "strategy1", "2": "strategy2",
                    "3": "strategy1", "4": "strategy2",
                    "5": "strategy3", "6": "strategy2",
                    "7": "strategy1",
                    "8": "strategy1", "9": "strategy2"}

initiLconfig = {
    "name": "AI Trust Game",
    "nRounds": 1,
    "nRoundsIsKnown": "True",
    "templateFilename": "ai_trust_game_four_pop",
    "llm": "OpenAIGPT4Turbo",
    "languages": [
        "en"
    ],
    "allAgentPermutations": "True",
    "agents": {
        "names": [
            "regulator",
            "developer",
            "user"
        ],
        "personalities": {
            "en": [
                "none"
            ]
        },
        "opponentPersonalityProb": [
            0,
            0,
            0
        ]
    }}

fairgame_configs = build_fairgame_configs({
    "result_payoffs": results["payoffs"],
    "strategy_dict": strategy_dict,
    "strategy_mapping": strategy_mapping,
    "fairgame_initial_config": initiLconfig,
})

data_utils.save_data({"_params": params},   
                     data_dir=f"data/fairgame_configs/{simulation_id}")
for idx, fairgame_config in enumerate(fairgame_configs):
    data_utils.save_data({f"FAIRGAME_config_ai_trust_four_pop_{idx}": fairgame_config},
                         data_dir=f"data/fairgame_configs/{simulation_id}")