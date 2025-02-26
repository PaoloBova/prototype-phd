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
    for k1, v1 in result_payoffs.items():
        combinations_dict[k1] = [
            [strategy_mapping[f"{utils.string_to_tuple(k1)[::-1][i]}"], f"weight_{k1}_{k2}"]
            for i, k2 in enumerate(v1.keys())
        ]
    
    fairgame_configs = []
    # Create a fairgame_config per payoff matrix index
    for idx in range(n_matrices):
        weights_dict = {}
        for k1, v1 in result_payoffs.items():
            for k2, v2 in v1.items():
                # If v2 is a numpy array, select the value at index 'idx'
                if isinstance(v2, numpy.ndarray):
                    if len(v2) > idx:
                        weight_val = float(v2[idx])
                    else:
                        logging.error(f"Index {idx} out of bounds for combination {k1}->{k2}")
                        raise IndexError
                else:
                    weight_val = float(v2)  # integer or float case
                weights_dict[f"weight_{k1}_{k2}"] = weight_val

            payoff_matrix = {
            "strategies": strategy_dict,
            "weights": weights_dict,
            "combinations": combinations_dict,
        }
        fairgame_config = {**initial_config, "payoffMatrix": payoff_matrix}
        fairgame_configs.append(fairgame_config)
    
    return fairgame_configs

simulation_id, current_commit, data_dir, plots_dir = data_utils.setup_project()

params = {**models.build_ai_trust(Eps={"start": -1, "stop": 1, "step": 0.1},
                                  β=1,
                                  cR=0.5,
                           ),
          "simulation_id": simulation_id,
          "commit": current_commit,
          "dispatch-type": 'multiple-populations',
          "payoffs_key": "ai-trust-v1",
          "Z": {"S3": 100, "S2": 100, "S1": 100},
          "allowed_sectors": {"P3": ["S3"],
                              "P2": ["S2"],
                              "P1": ["S1"], },
          "sector_strategies": {"S3": [5, 7],
                                "S2": [3, 4],
                                "S1": [1, 2], },
          }

results = utils.thread_macro(params,
                       model_utils.create_profiles,
                       model_utils.apply_profile_filters,
                       payoffs.build_payoffs,
                       )

strategy_dict = {"en": {"strategy1": "Defect",
                        "strategy2": "Cooperate"}}
strategy_mapping = {"1": "strategy1", "2": "strategy2",
                    "3": "strategy1", "4": "strategy2",
                    "5": "strategy1", "6": "strategy2",
                    "7": "strategy3"}

initiLconfig = {
    "name": "AI Trust Game",
    "nRounds": 1,
    "nRoundsIsKnown": "True",
    "templateFilename": "ai_trust_game",
    "llm": "OpenAIGPT4Turbo",
    "languages": [
        "en"
    ],
    "allAgentPermutations": "False",
    "agents": {
        "names": [
            "regulator1",
            "developer1",
            "user1"
        ],
        "personalities": {
            "en": [
                "cooperative",
                "aggressive",
                "aggressive"
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
    data_utils.save_data({f"FAIRGAME_config_ai_trust_v1_{idx}": fairgame_config},
                         data_dir=f"data/fairgame_configs/{simulation_id}")