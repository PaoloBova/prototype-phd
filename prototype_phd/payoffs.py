from .utils import *
from .types import *
from .methods import *
from .model_utils import *

import collections
import numpy as np
import typing

valid_dtypes = typing.Union[float, list[float], np.ndarray, dict]
def build_DSAIR(b:valid_dtypes=4, # benefit: The size of the per round benefit of leading the AI development race, b>0
                c:valid_dtypes=1, # cost: The cost of implementing safety recommendations per round, c>0
                s:valid_dtypes={"start":1, # speed: The speed advantage from choosing to ignore safety recommendations, s>1
                                "stop":5.1,
                                "step":0.1}, 
                p:valid_dtypes={"start":0, # avoid_risk: The probability that unsafe firms avoid an AI disaster, p ∈ [0, 1]
                                "stop":1.02,
                                "step":0.02}, 
                B:valid_dtypes=10**4, # prize: The size of the prize from winning the AI development race, B>>b
                W:valid_dtypes=100, # timeline: The anticipated timeline until the development race has a winner if everyone behaves safely, W ∈ [10, 10**6]
                pfo:valid_dtypes=0, # detection risk: The probability that firms who ignore safety precautions are found out, pfo ∈ [0, 1]
                α:valid_dtypes=0, # the cost of rewarding/punishing a peer
                γ:valid_dtypes=0, # the effect of a reward/punishment on a developer's speed
                epsilon:valid_dtypes=0, # commitment_cost: The cost of setting up and maintaining a voluntary commitment, ϵ > 0
                ω:valid_dtypes=0, # noise: Noise in arranging an agreement, with some probability they fail to succeed in making an agreement, ω ∈ [0, 1]
                collective_risk:valid_dtypes=0, # The likelihood that a disaster affects all actors
                β:valid_dtypes=0.01, # learning_rate: the rate at which players imitate each other
                Z:int=100, # population_size: the number of players in the evolutionary game
                strategy_set:list[str]=["AS", "AU"], # the set of available strategies
                exclude_args:list[str]=['Z', 'strategy_set'], # a list of arguments that should be returned as they are
                override:bool=False, # whether to build the grid if it is very large
                drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
               ) -> dict: # A dictionary containing items from `ModelTypeDSAIR` and `ModelTypeEGT`
    """Initialise baseline DSAIR models for all combinations of the provided
    parameter valules. By default, we create models for replicating Figure 1
    of Han et al. 2021."""
    
    saved_args = locals()
    models = model_builder(saved_args,
                           exclude_args=exclude_args,
                           override=override,
                           drop_args=drop_args)
    return models

def payoffs_sr(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs_sr`
    """The short run payoffs for the DSAIR game."""
    s, b, c = [models[k] for k in ['s', 'b', 'c']]
    πAA = -c + b/2
    πAB = -c + b/(s+1)
    πBA = s*b/(s+1)
    πBB = b/2
    
    # Promote all stacks to 3D arrays
    πAA = πAA[:, None, None]
    πAB = πAB[:, None, None]
    πBA = πBA[:, None, None]
    πBB = πBB[:, None, None]
    matrix = np.block([[πAA, πAB], 
                       [πBA, πBB]])
    return {**models, 'payoffs_sr':matrix}

def payoffs_sr_pfo_extension(models):
    """The short run payoffs for the DSAIR game with a chance of unsafe
    behaviour being spotted."""
    s, b, c, pfo = [models[k] for k in ['s', 'b', 'c', 'pfo']]
    πAA = -c + b/2
    πAB = -c + b/(s+1) * (1 - pfo) + pfo * b
    πBA = (1 - pfo) * s * b / (s+1)
    πBB = (1 - pfo**2) * b/2
    
    # Promote all stacks to 3D arrays
    πAA = πAA[:, None, None]
    πAB = πAB[:, None, None]
    πBA = πBA[:, None, None]
    πBB = πBB[:, None, None]
    matrix = np.block([[πAA, πAB],
                       [πBA, πBB]])
    return {**models, 'payoffs_sr':matrix}

def payoffs_lr(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`
    """The long run average payoffs for the DSAIR game."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s, p, B, W = [models[k][:, None, None]
                  for k in ['s', 'p', 'B', 'W']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]    
    πAA = πAA + B/(2*W)
    πAB = πAB
    πBA = p*(s*B/W + πBA)
    πBB = p*(s*B/(2*W) + πBB)
    payoffs = np.block([[πAA, πAB],
                        [πBA, πBB]])
    return {**models, 'payoffs': payoffs}

def punished_and_sanctioned_payoffs(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
                                   ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """Compute the payoffs for the punished and sanctioner players in a DSAIR
    model with peer punishment."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W, pfo = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W', 'pfo']]
    α, γ = [models[k][:, None, None] for k in ['α', 'γ']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    
    s_punished = s - γ
    s_sanctioner = 1 - α
    sum_of_speeds = np.maximum(1e-20, s_punished + s_sanctioner)
    punished_wins = (s_punished > 0) & (((W-s)*np.maximum(0, s_sanctioner))
                                        <= ((W-1) * s_punished))
    punished_draws = (s_punished > 0) & (((W-s) * s_sanctioner)
                                         == ((W-1) * s_punished))
    sanctioner_wins = (s_sanctioner > 0) & (((W-s) * s_sanctioner)
                                            >= ((W-1)*np.maximum(0,s_punished)))
    no_winner = (s_punished <= 0) & (s_sanctioner <= 0)

    both_speeds_positive = (s_punished > 0) & (s_sanctioner > 0)
    only_sanctioner_speed_positive = (s_punished <= 0) & (s_sanctioner > 0)
    only_punisher_speed_positive = (s_punished > 0) & (s_sanctioner <= 0)

    p_loss = np.where(punished_wins | punished_draws, p, 1)
    R = np.where(no_winner,
                 1e50,
                 1 + np.minimum((W-s)/ np.maximum(s_punished, 1e-10),
                                (W-1)/ np.maximum(s_sanctioner, 1e-10)))
    B_s = np.where(sanctioner_wins, B, np.where(punished_draws, B/2, 0))
    B_p = np.where(punished_wins, B, np.where(punished_draws, B/2, 0))
    b_s = np.where(both_speeds_positive,
                   (1-pfo) * b * s_sanctioner / sum_of_speeds + pfo * b,
                   np.where(only_sanctioner_speed_positive, b, 0))
    b_p = np.where(both_speeds_positive,
                   (1-pfo) * b * s_punished / sum_of_speeds,
                   np.where(only_punisher_speed_positive, (1 - pfo)*b, 0))
    sanctioner_payoff = (1 / R) * (πAB + B_s - (b_s - c)) + (b_s - c)
    # sanctioner_payoff = (1 / R) * (πAB + B_s + (R-1)*(b_s - c))
    punished_payoff = (p_loss / R) * (πBA + B_p - b_p) + p_loss * b_p
    # punished_payoff = (p_loss / R) * (πBA + B_p + (R-1)*b_p)
    return {**models,
            'sanctioner_payoff':sanctioner_payoff,
            'punished_payoff':punished_payoff}

def payoffs_lr_peer_punishment(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """The long run average payoffs for the DSAIR game with peer punishment."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W']]
    α, γ = [models[k][:, None, None] for k in ['α', 'γ']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    models = punished_and_sanctioned_payoffs(models)
    
    ΠAA = πAA + B/(2*W)
    ΠAB = πAB
    ΠAC = πAA + B/(2*W)
    ΠBA = p*(s*B/W + πBA)
    ΠBB = p*(s*B/(2*W) + πBB)
    ΠBC = models["punished_payoff"]
    ΠCA = πAA + B/(2*W)
    ΠCB = models["sanctioner_payoff"]
    ΠCC = πAA + B/(2*W)
    matrix = np.block([[ΠAA, ΠAB, ΠAC], 
                       [ΠBA, ΠBB, ΠBC],
                       [ΠCA, ΠCB, ΠCC],
                       ])
    return {**models, 'payoffs':matrix}

def payoffs_lr_peer_reward(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """The long run average payoffs for the DSAIR game with peer punishment."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W']]
    α, γ = [models[k][:, None, None] for k in ['α', 'γ']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    
    s_rewarded = 1 + γ
    s_helper = np.maximum(0, 1 - α)
    s_colaborative = np.maximum(0, 1 + γ - α)
    ΠAA = πAA + B/(2*W)
    ΠAB = πBA
    ΠAC = πAA + B * s_rewarded / W
    ΠBA = p*(s*B/W + πBA)
    ΠBB = p*(s*B/(2*W) + πBB)
    ΠBC = p*(s*B/W + πBA)
    ΠCA = πAA
    ΠCB = πAB
    ΠCC = πAA + B * s_colaborative/(2*W)
    matrix = np.block([[ΠAA, ΠAB, ΠAC], 
                       [ΠBA, ΠBB, ΠBC],
                       [ΠCA, ΠCB, ΠCC],
                       ])
    return {**models, 'payoffs':matrix}

def payoffs_lr_voluntary(models:dict, # A dictionary containing the items in `ModelTypeDSAIR`
              ) -> dict : # The `models` dictionary with added payoff matrix `payoffs`:
    """The long run average payoffs for the DSAIR game with voluntary
    commitments."""
    # All 1D arrays must be promoted to 3D Arrays for broadcasting
    s,b,c, p, B, W = [models[k][:, None, None]
                      for k in ['s', 'b', 'c', 'p', 'B', 'W']]
    α, γ, ϵ = [models[k][:, None, None] for k in ['α', 'γ', 'epsilon']]
    πAA,πAB,πBA,πBB = [models['payoffs_sr'][:, i:i+1, j:j+1]
                       for i in range(2) for j in range(2)]
    models = punished_and_sanctioned_payoffs(models)
    
    ΠAA = πAA + B/(2*W)
    ΠAB = πAB
    ΠAC = πAB
    ΠAD = πAB
    ΠAE = πAB
    ΠBA = p*(s*B/W + πBA)
    ΠBB = p*(s*B/(2*W) + πBB)
    ΠBC = p*(s*B/(2*W) + πBB)
    ΠBD = p*(s*B/(2*W) + πBB)
    ΠBE = p*(s*B/(2*W) + πBB)
    ΠCA = p*(s*B/W + πBA)
    ΠCB = p*(s*B/(2*W) + πBB)
    ΠCC = πAA + B/(2*W) - ϵ
    ΠCD = πAB - ϵ
    ΠCE = πAA + B/(2*W) - ϵ
    ΠDA = p*(s*B/W + πBA)
    ΠDB = p*(s*B/(2*W) + πBB)
    ΠDC = p*(s*B/W + πBA) - ϵ
    ΠDD = p*(s*B/(2*W) + πBB) - ϵ
    ΠDE = models['punished_payoff'] - ϵ
    ΠEA = p*(s*B/W + πBA) - ϵ
    ΠEB = p*(s*B/(2*W) + πBB)
    ΠEC = πAA + B/(2*W) - ϵ
    ΠED = models['sanctioner_payoff'] - ϵ
    ΠEE = πAA + B/(2*W) - ϵ
    matrix = np.block([[ΠAA, ΠAB, ΠAC, ΠAD, ΠAE], 
                       [ΠBA, ΠBB, ΠBC, ΠBD, ΠBE],
                       [ΠCA, ΠCB, ΠCC, ΠCD, ΠCE],
                       [ΠDA, ΠDB, ΠDC, ΠDD, ΠDE],
                       [ΠEA, ΠEB, ΠEC, ΠED, ΠEE]
                       ])
    return {**models, 'payoffs':matrix}

def payoffs_encanacao_2016(models):
    names = ['b_r', 'b_s', 'c_s', 'c_t', 'σ']
    b_r, b_s, c_s, c_t, σ = [models[k] for k in names]
    payoffs = {}
    n_players = 3
    n_sectors = 3
    n_strategies_per_sector = [2, 2, 2]
    n_strategies_total = 6
    # All players are from the first sector, playing that sector's first strategy
    index_min = "0-0-0"
    # All players are from the third sector, playing that sector's second strategy
    index_max = "5-5-5"
    # Note: The seperator makes it easy to represent games where n_strategies_total >= 10.

    # It is also trivial to define a vector which maps these indexes to strategy profiles
    # As sector order is fixed we could neglect to mention suscripts for each sector
    strategy_names = ["D", "C", "D", "C", "D", "C"]

    zero = np.zeros(b_r.shape[0])
    # As in the main text
    payoffs["C-C-C"] = {"P3": b_r-2*c_s,
                        "P2": σ+b_s-c_t,
                        "P1": σ+b_s}
    payoffs["C-C-D"] = {"P3": -c_s,
                        "P2": b_s-c_t,
                        "P1": zero}
    payoffs["C-D-C"] = {"P3": b_r-c_s,
                        "P2": zero,
                        "P1": b_s}
    payoffs["C-D-D"] = {"P3": zero,
                        "P2": σ,
                        "P1": σ}
    payoffs["D-C-C"] = {"P3": zero,
                        "P2": σ-c_t,
                        "P1": σ}
    payoffs["D-C-D"] = {"P3": zero,
                        "P2": -c_t,
                        "P1": zero}
    payoffs["D-D-C"] = {"P3": zero,
                        "P2": zero,
                        "P1": zero}
    payoffs["D-D-D"] = {"P3": zero,
                        "P2": σ,
                        "P1": σ}

    # The following indexes capture all strategy profiles where each player is fixed to a unique sector
    # (and player order does not matter, so we need only consider one ordering of sectors).
    payoffs["4-2-0"] = payoffs["D-D-D"]
    payoffs["4-2-1"] = payoffs["D-D-C"]
    payoffs["4-3-0"] = payoffs["D-C-D"]
    payoffs["4-3-1"] = payoffs["D-C-C"]
    payoffs["5-2-0"] = payoffs["C-D-D"]
    payoffs["5-2-1"] = payoffs["C-D-C"]
    payoffs["5-3-0"] = payoffs["C-C-D"]
    payoffs["5-3-1"] = payoffs["C-C-C"]
    return {**models, "payoffs": payoffs}

@multi
def build_payoffs(models: dict):
    return models.get('payoffs_key')

@method(build_payoffs, 'vasconcelos_2014_primitives')
def build_payoffs(models: dict):
    names = ['payoffs_state', 'c', 'T', 'b_r', 'b_p', 'r']
    payoffs_state, c, T, b_r, b_p, r = [models[k] for k in names]
    strategy_counts = payoffs_state['strategy_counts']
    n_r = strategy_counts["2"]
    n_p = strategy_counts["4"]
    risk = r * (n_r * c * b_r + n_p * c * b_p < T)
    # The payoffs must be computed for each strategy type in the interaction.
    # In games where we employ hypergeometric sampling, we usually do not
    # care about player order in the interaction. If order did matter, then
    # we would represent the payoffs per strategy still but it would capture
    # the expected payoffs given how likely a player of that strategy was to
    # play in each node of the extensive-form game. Non-players of type 0
    # usually do not have payoffs.
    payoffs = {"1": (1 - risk) * b_r,  # rich_free_rider
               "2": (1 - risk) * c * b_r,  # rich_contributor
               "3": (1 - risk) * b_p,  # poor_free_rider
               "4": (1 - risk) * c * b_p}  # poor_contributor
    return {**models, "payoff_primitives": payoffs}

@method(build_payoffs, 'vasconcelos_2014')
def build_payoffs(models: dict):
    profiles = create_profiles({'n_players': models.get('n_players', 5),
                                'n_strategies': [2, 2]})['profiles']
    payoffs = {}
    for profile in profiles:
        profile_tuple = thread_macro(profile,
                                     (str.split, "-"),
                                     (map, int, "self"),
                                     list,
                                     reversed,
                                     list,
                                     np.array,
                                     )
        strategy_counts = {f"{i}": np.sum(
            profile_tuple == i) for i in range(5)}
        payoffs_state = {'strategy_counts': strategy_counts}
        primitives = thread_macro(models,
                                  (assoc,
                                   'payoffs_state', payoffs_state,
                                   'payoffs_key', "vasconcelos_2014_primitives"),
                                  build_payoffs,
                                  (get, "payoff_primitives"),
                                  )
        payoffs[profile] = {}
        for i, strategy in enumerate(profile_tuple):
            if strategy == 0:
                continue
            elif strategy == 1:
                payoffs[profile][f"P{i+1}"] = primitives['1']
            elif strategy == 2:
                payoffs[profile][f"P{i+1}"] = primitives['2']
            elif strategy == 3:
                payoffs[profile][f"P{i+1}"] = primitives['3']
            elif strategy == 4:
                payoffs[profile][f"P{i+1}"] = primitives['4']
            else:
                continue
    return {**models, "payoffs": payoffs}

@method(build_payoffs, 'payoff_function_wrapper')
def build_payoffs(models: dict):
    profiles = create_profiles(models)['profiles']
    profile_payoffs_key = models['profile_payoffs_key']
    payoffs_state = models.get("payoffs_state", {})
    payoffs = {}
    for profile in profiles:
        profile_tuple = string_to_tuple(profile)
        strategy_counts = dict(zip(*np.unique(profile_tuple,
                                              return_counts=True)))
        payoffs_state = {**payoffs_state,
                         'strategy_counts': strategy_counts}
        profile_models = {**models,
                          "strategy_profile": profile,
                          "payoffs_state": payoffs_state,
                          "payoffs_key": profile_payoffs_key}
        profile_payoffs = thread_macro(profile_models,
                                       build_payoffs,
                                       (get, "profile_payoffs"),
                                       )
        payoffs[profile] = {}
        for i, strategy in enumerate(profile_tuple):
            if strategy == 0:
                # A strategy of 0 is reserved for missing players, missing
                # players do not have payoffs.
                continue
            elif str(strategy) in profile_payoffs.keys():
                payoffs[profile][f"P{i+1}"] = profile_payoffs[f"{strategy}"]
            else:
                continue
    return {**models, "payoffs": payoffs}

@method(build_payoffs, "flow_payoffs_wrapper")
def build_payoffs(models):
    "Build the flow payoffs for each state-action in a stochastic game."
    state_actions = models['state_actions']
    payoffs_state = models.get('payoffs_state', {})
    flow_payoffs = collections.defaultdict()
    for state_action in state_actions:
        state, action_profile = str.split(state_action, ":")
        action_tuple = string_to_tuple(action_profile)
        action_counts = dict(zip(*np.unique(action_tuple,
                                            return_counts=True)))
        payoffs_state = {**payoffs_state,
                         'strategy_counts': action_counts,
                         'state': state}
        payoffs_flow_key = models['payoffs_flow_key']
        profile_models = {**models,
                          "payoffs_state": payoffs_state,
                          "payoffs_key": payoffs_flow_key}
        flow_payoffs[state_action] = thread_macro(profile_models,
                                                  build_payoffs,
                                                  (get, "flow_payoffs"),
                                                  )
    return {**models, "flow_payoffs": flow_payoffs}

@multi
def compute_transition(models):
    "Compute the transition likelihood for the given transition."
    return models.get('compute_transition_key')

@method(compute_transition, 'anonymous_actions')
def compute_transition(models):
    """Compute transition likelihood when we are only passed anonymous action
    profiles (i.e. order does not matter)."""
    P, Q = [models[k] for k in ['P', 'Q']]
    transition_start, transition_end = [models[k] for k in ['transition_start',
                                                            'transition_end']]
    next_state, action_profile = transition_end.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    profiles = create_profiles({**models,
                                "profiles_rule": "from_strategy_count",
                                "strategy_count": action_count})['profiles']
    profile_tuples = map(string_to_tuple, profiles)
    p = [np.prod([P[f"P{player + 1}"][transition_start].get(f"A{action}", 0)
                  for player, action in enumerate(profile_tuple)])
         for profile_tuple in profile_tuples]
    return np.sum(p) * Q[transition_start][next_state]

@method(compute_transition)
def compute_transition(models):
    "Compute transition likelihood given the states and action profiles."
    P, Q = [models[k] for k in ['P', 'Q']]
    transition_start, transition_end = [models[k] for k in ['transition_start',
                                                            'transition_end']]
    next_state, action_profile = transition_end.split(":")
    action_tuple = string_to_tuple(action_profile)
    p = np.prod([P[f"P{player + 1}"][transition_start].get(f"A{action}", 0)
                 for player, action in enumerate(action_tuple)])
    return p * Q[transition_start][next_state]

@method(build_payoffs, "stochastic-no-discounting")
def build_payoffs(models: dict):
    """Compute the payoffs for a stochastic game with the given flow_payoffs,
    state_transitions, strategies, and strategy_profile, when there is no
    discounting."""
    u = models['flow_payoffs']
    Q = models['state_transitions']
    strategy_profile = models['strategy_profile'].split("-")[::-1]
    strategies = models['strategies']
    P = {f"P{player + 1}": strategies[strategy_key]
         for player, strategy_key in enumerate(strategy_profile)}
    state_actions = list(Q.keys())
    M = np.zeros((len(state_actions), len(state_actions)))
    for row, transition_start in enumerate(state_actions):
        for col, transition_end in enumerate(state_actions):
            transition_data = {**models,
                               "P": P,
                               "Q": Q,
                               "transition_start": transition_start,
                               "transition_end": transition_end}
            M[row, col] = compute_transition(transition_data)
    v = thread_macro({**models, "transition_matrix": np.array([M])},
                     find_ergodic_distribution,
                     (get, "ergodic"))[0]
    u = np.array([[u[s][f"{i+1}"] for i in range(len(u[s]))]
                  for s in state_actions])
    for _ in range(v.ndim, u.ndim):
        v = v[:, None]
    payoffs = np.sum(v * u, axis=0)
    profile_payoffs = {f"{i+1}": pi for i, pi in enumerate(payoffs)}
    return {**models, "profile_payoffs": profile_payoffs}

@method(build_payoffs, "stochastic-with-discounting")
def build_payoffs(models: dict):
    """Compute the payoffs for a stochastic game with the given flow_payoffs,
    state_transitions, strategies, and strategy_profile."""
    u = models['flow_payoffs']
    Q = models['state_transitions']
    d = models['discount_rate']
    v0 = models['initial_state_action_distribution']
    strategy_profile = models['strategy_profile'].split("-")[::-1]
    strategies = models['strategies']
    P = {f"P{player + 1}": strategies[strategy_key]
         for player, strategy_key in enumerate(strategy_profile)}
    state_actions = list(Q.keys())
    M = np.zeros((len(state_actions), len(state_actions)))
    for row, transition_start in enumerate(state_actions):
        for col, transition_end in enumerate(state_actions):
            transition_data = {**models,
                               "P": P,
                               "Q": Q,
                               "transition_start": transition_start,
                               "transition_end": transition_end}
            M[row, col] = compute_transition(transition_data)
    v = (1 - d) * v0 * np.linalg.inv(np.eye(M.shape) - d * M)
    u = np.array([[u[s][f"{i+1}"] for i in range(len(u[s]))]
                  for s in state_actions])
    for _ in range(v.ndim, u.ndim):
        v = v[:, None]
    payoffs = np.sum(v * u, axis=0)
    profile_payoffs = {f"{i+1}": pi for i, pi in enumerate(payoffs)}
    return {**models, "profile_payoffs": profile_payoffs}

@method(build_payoffs, 'vasconcelos_2014_flow')
def build_payoffs(models: dict):
    names = ['payoffs_state', 'c', 'T', 'b_r', 'b_p', 'r', 'g']
    payoffs_state, c, T, b_r, b_p, r, g = [models[k] for k in names]
    strategy_counts = payoffs_state['strategy_counts']
    state = payoffs_state['state']
    reward_bonus = g if state=='1' else 1
    n_r = strategy_counts.get("2", 0)
    n_p = strategy_counts.get("4", 0)
    risk = r * (n_r * c * b_r + n_p * c * b_p < T)
    payoffs = {"1": (1 - risk) * b_r * reward_bonus,  # rich_free_rider
               "2": (1 - risk) * c * b_r * reward_bonus,  # rich_contributor
               "3": (1 - risk) * b_p * reward_bonus,  # poor_free_rider
               "4": (1 - risk) * c * b_p * reward_bonus}  # poor_contributor
    return {**models, "flow_payoffs": payoffs}

@multi
def state_transition(models):
    "Compute the likelihood of the given state_transition."
    return models.get('state_transition_key')

@method(state_transition, 'ex1')
def state_transition(models):
    """Compute transition likelihood for a model with 2 states and an arbitrary
    number of players. To stay in the good state, 0, all players need to choose
    to cooperate, i.e. action 1."""
    state_action, next_state = [models[k] for k in ['state_action',
                                                    'next_state']]
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and next_state == '1'
        and n_cooperators != n_players):
        transition_likelihood = 1
    elif (current_state == '1'
          and next_state == '0'
          and n_cooperators == n_players):
        transition_likelihood = 1
    elif (current_state == '0'
          and next_state == '0'
          and n_cooperators == n_players):
        transition_likelihood = 1
    elif (current_state == '1'
          and next_state == '1'
          and n_cooperators != n_players):
        transition_likelihood = 1
    else:
        transition_likelihood = 0
    return transition_likelihood

def build_state_transitions(models):
    state_actions = models['state_actions']
    n_states = models['n_states']
    state_transitions = {}
    for state_action in state_actions:
        state_transitions[state_action] = {}
        for next_state in [f"{i}" for i in range(n_states)]:
            likelihood = state_transition({**models,
                                           "state_action": state_action,
                                           "next_state": next_state})
            state_transitions[state_action][next_state] = likelihood
    return {**models, "state_transitions": state_transitions}

@multi
def build_strategy(models):
    "Build the desired strategy"
    return models.get('strategy_key')

@method(build_strategy, 'ex1_rich_cooperator')
def build_strategy(models):
    """A rich player who cooperates with 95% probability if everyone currently
    cooperates, otherwise defects with 95% probability."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A1": 0.95, "A2": 0.05}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A1": 0.95, "A2": 0.05}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    return strategy

@method(build_strategy, 'ex1_rich_defector')
def build_strategy(models):
    """A rich player who defects with 95% probability no matter what others
    do, nor what state they are in."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A1": 0.05, "A2": 0.95}
    return strategy

@method(build_strategy, 'ex1_poor_cooperator')
def build_strategy(models):
    """A poor player who cooperates with 95% probability if everyone currently
    cooperates, otherwise defects with 95% probability."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A3": 0.95, "A4": 0.05}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A3": 0.95, "A4": 0.05}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    return strategy

@method(build_strategy, 'ex1_poor_defector')
def build_strategy(models):
    """A poor player who defects with 95% probability no matter what others
    do, nor what state they are in."""
    state_action = models['state_action']
    current_state, action_profile = state_action.split(":")
    action_tuple = string_to_tuple(action_profile)
    action_count = dict(zip(*np.unique(action_tuple, return_counts=True)))
    n_players = len(action_tuple)
    n_cooperators = action_count.get(1, 0) + action_count.get(3, 0)
    if (current_state == '0'
        and n_cooperators == n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '0'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '1'
          and n_cooperators == n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    elif (current_state == '1'
          and n_cooperators != n_players):
        strategy = {"A3": 0.05, "A4": 0.95}
    return strategy

def build_strategies(models):
    "Build a dictionary containing the specified strategies in `models`"
    state_actions, strategy_keys = [models[k] for k in ["state_actions",
                                                        "strategy_keys"]]
    strategies = {f"{i+1}": {s: build_strategy({"strategy_key": strategy_key,
                                            "state_action": s})
                         for s in state_actions}
              for i, strategy_key in enumerate(strategy_keys)}
    return {**models, "strategies": strategies}

@method(build_payoffs, "regulatory_markets_v1_reward_before")
def build_payoffs(models):
    """Regulatory market payoffs when incentives are given in advance and only
    taken away if firms act unsafely."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = b / (s+1) * (1 - pfo_h) * risk_shared + (b + B / W) * pfo_h - c
    Π_h21 = p * ( 1 - pfo_h) * (s*b / (s + 1) + s * B / W)
    Π_h22 = p * ( 1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = b / (s+1) * (1 - pfo_l) * risk_shared + (b + B / W) * pfo_l - c
    Π_l21 = p * ( 1 - pfo_l) * (s*b / (s + 1) + s * B / W)
    Π_l22 = p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g
    Ω_12 = r_l + g
    Ω_21 = r_h + g * pfo_h**2 - λ_h
    Ω_22 = r_l + g * pfo_l**2 - λ_l
    Ω_31 = r_h + g
    Ω_32 = r_l + g * pfo_l**2 - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v1_reward_after")
def build_payoffs(models):
    """Regulatory market payoffs when there is only a reward after catching
    unsafe firms."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = b / (s+1) * (1 - pfo_h) * risk_shared + (b + B / W) * pfo_h - c
    Π_h21 = p * ( 1 - pfo_h) * (s*b / (s + 1) + s * B / W)
    Π_h22 = p * ( 1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = b / (s+1) * (1 - pfo_l) * risk_shared + (b + B / W) * pfo_l - c
    Π_l21 = p * ( 1 - pfo_l) * (s*b / (s + 1) + s * B / W)
    Π_l22 = p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    # No ex-ante reward for regulators
    Ω_11 = r_h
    Ω_12 = r_l
    # Expect to catch n*p unsafe firms, where n=2 and p=pfo_h
    # They may expect to be penalised if a disaster occurs under their watch
    # but by default the penalty λ may be 0.
    Ω_21 = r_h + g * 2 * pfo_h - λ_h
    Ω_22 = r_l + g * 2 * pfo_l - λ_l
    Ω_31 = r_h
    Ω_32 = r_l + g * 2 * pfo_l - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v1a_reward_after")
def build_payoffs(models):
    """An alternative payoff scheme for regulatory markets which more closely
    matches the DSAIR model"""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = b / (s+1) * (1 - pfo_h) * risk_shared + (b) * pfo_h - c
    Π_h21 = p * (s*b / (s + 1)  * ( 1 - pfo_h) + s * B / W)
    Π_h22 = p * (b/2 * ( 1 - pfo_h**2) + s*B/(2*W)) * risk_shared
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = b / (s+1) * (1 - pfo_l) * risk_shared + (b) * pfo_l - c
    Π_l21 = p * (s*b / (s + 1) * ( 1 - pfo_l) + s * B / W)
    Π_l22 = p * (b/2 * ( 1 - pfo_l**2) + s*B/(2*W)) * risk_shared
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    # No ex-ante reward for regulators
    Ω_11 = r_h
    Ω_12 = r_l
    # Expect to catch n*p unsafe firms, where n=2 and p=pfo_h
    # They may expect to be penalised if a disaster occurs under their watch
    # but by default the penalty λ may be 0.
    Ω_21 = r_h + g * 2 * pfo_h - λ_h
    Ω_22 = r_l + g * 2 * pfo_l - λ_l
    Ω_31 = r_h
    Ω_32 = r_l + g * 2 * pfo_l - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v1a_reward_before")
def build_payoffs(models):
    """Regulatory market payoffs when incentives are given in advance and only
    taken away if firms act unsafely."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = b / (s+1) * (1 - pfo_h) * risk_shared + (b) * pfo_h - c
    Π_h21 = p * (s*b / (s + 1)  * ( 1 - pfo_h) + s * B / W)
    Π_h22 = p * (b/2 * ( 1 - pfo_h**2) + s*B/(2*W)) * risk_shared
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = b / (s+1) * (1 - pfo_l) * risk_shared + (b) * pfo_l - c
    Π_l21 = p * (s*b / (s + 1) * ( 1 - pfo_l) + s * B / W)
    Π_l22 = p * (b/2 * ( 1 - pfo_l**2) + s*B/(2*W)) * risk_shared
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g
    Ω_12 = r_l + g
    Ω_21 = r_h + g * pfo_h**2 - λ_h
    Ω_22 = r_l + g * pfo_l**2 - λ_l
    Ω_31 = r_h + g
    Ω_32 = r_l + g * pfo_l**2 - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v1_reward_mixed")
def build_payoffs(models):
    """Regulatory market payoffs when there is only a reward after catching
    unsafe firms."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    mix = models.get('incentive_mix', 0)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = b / (s+1) * (1 - pfo_h) * risk_shared + (b + B / W) * pfo_h - c
    Π_h21 = p * ( 1 - pfo_h) * (s*b / (s + 1) + s * B / W)
    Π_h22 = p * ( 1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = b / (s+1) * (1 - pfo_l) * risk_shared + (b + B / W) * pfo_l - c
    Π_l21 = p * ( 1 - pfo_l) * (s*b / (s + 1) + s * B / W)
    Π_l22 = p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g * mix
    Ω_12 = r_l + g * mix
    Ω_21 = r_h + g * (pfo_h**2 * mix + pfo_h * (1 - mix)) - λ_h
    Ω_22 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    Ω_31 = r_h + g * mix
    Ω_32 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@multi
def compute_game_welfare(models):
    "Compute the welfare generated by the game in each state."
    return models.get('game_welfare_rule')

@method(compute_game_welfare, "regulatory_markets_v1_reward_before")
def compute_game_welfare(models):
    "Compute the welfare generated by the game in each state."
    names = ['payoffs', 'ergodic']
    payoffs, ergodic = [models[k] for k in names]
    p, g, pfo_h, pfo_l = [models[k] for k in ['p', 'g', 'pfo_h', 'pfo_l']]
    consumer_surplus = models.get('consumer_surplus', 0)
    externality = models.get('externality', 0)

    company_payoffs_safe_hq = payoffs['4-1-1']["P1"] + payoffs['4-1-1']["P2"]
    company_payoffs_safe_lq = payoffs['5-1-1']["P1"] + payoffs['5-1-1']["P2"]
    company_payoffs_unsafe_hq = payoffs['4-2-2']["P1"] + payoffs['4-2-2']["P2"]
    company_payoffs_unsafe_lq = payoffs['5-2-2']["P1"] + payoffs['5-2-2']["P2"]
    company_payoffs_vetted_hq = payoffs['4-3-3']["P1"] + payoffs['4-3-3']["P2"]
    company_payoffs_vetted_lq = payoffs['5-3-3']["P1"] + payoffs['5-3-3']["P2"]

    welfare_safe_hq = (company_payoffs_safe_hq * (1 + consumer_surplus)
                       - g)
    welfare_unsafe_hq = (company_payoffs_unsafe_hq * (1 + consumer_surplus)
                         - (1-p)*externality
                         - g * pfo_h**2)
    welfare_vetted_hq = (company_payoffs_vetted_hq * (1 + consumer_surplus)
                         - g)
    welfare_safe_lq = (company_payoffs_safe_lq * (1 + consumer_surplus)
                       - g)
    welfare_unsafe_lq = (company_payoffs_unsafe_lq * (1 + consumer_surplus)
                       - (1-p)*externality
                       - g * pfo_l**2)
    welfare_vetted_lq = (company_payoffs_vetted_lq * (1 + consumer_surplus)
                       - (1-p)*externality
                       - g * pfo_l**2)
    welfares = [welfare_safe_hq,
                welfare_unsafe_hq,
                welfare_vetted_hq,
                welfare_safe_lq,
                welfare_unsafe_lq,
                welfare_vetted_lq]
    game_welfare = np.sum([welfare * state_frequency
                           for welfare, state_frequency in zip(welfares,
                                                               ergodic.T)],
                          axis=0)
    return {**models, "game_welfare": game_welfare}

@method(compute_game_welfare, "regulatory_markets_v1_reward_after")
def compute_game_welfare(models):
    "Compute the welfare generated by the game in each state."
    names = ['payoffs', 'ergodic']
    payoffs, ergodic = [models[k] for k in names]
    p, g, pfo_h, pfo_l = [models[k] for k in ['p', 'g', 'pfo_h', 'pfo_l']]
    consumer_surplus = models.get('consumer_surplus', 0)
    externality = models.get('externality', 0)

    company_payoffs_safe_hq = payoffs['4-1-1']["P1"] + payoffs['4-1-1']["P2"]
    company_payoffs_safe_lq = payoffs['5-1-1']["P1"] + payoffs['5-1-1']["P2"]
    company_payoffs_unsafe_hq = payoffs['4-2-2']["P1"] + payoffs['4-2-2']["P2"]
    company_payoffs_unsafe_lq = payoffs['5-2-2']["P1"] + payoffs['5-2-2']["P2"]
    company_payoffs_vetted_hq = payoffs['4-3-3']["P1"] + payoffs['4-3-3']["P2"]
    company_payoffs_vetted_lq = payoffs['5-3-3']["P1"] + payoffs['5-3-3']["P2"]

    welfare_safe_hq = (company_payoffs_safe_hq * (1 + consumer_surplus))
    welfare_unsafe_hq = (company_payoffs_unsafe_hq * (1 + consumer_surplus)
                         - (1-p)*externality
                         - g * pfo_h)
    welfare_vetted_hq = (company_payoffs_vetted_hq * (1 + consumer_surplus))
    welfare_safe_lq = (company_payoffs_safe_lq * (1 + consumer_surplus))
    welfare_unsafe_lq = (company_payoffs_unsafe_lq * (1 + consumer_surplus)
                         - (1-p)*externality
                         - g * pfo_l)
    welfare_vetted_lq = (company_payoffs_vetted_lq * (1 + consumer_surplus)
                         - (1-p)*externality
                         - g * pfo_l)
    welfares = [welfare_safe_hq,
                welfare_unsafe_hq,
                welfare_vetted_hq,
                welfare_safe_lq,
                welfare_unsafe_lq,
                welfare_vetted_lq]
    game_welfare = np.sum([welfare * state_frequency
                           for welfare, state_frequency in zip(welfares,
                                                               ergodic.T)],
                          axis=0)
    return {**models, "game_welfare": game_welfare}

@method(compute_game_welfare, "regulatory_markets_v1_reward_mixed")
def compute_game_welfare(models):
    "Compute the welfare generated by the game in each state."
    names = ['payoffs', 'ergodic']
    payoffs, ergodic = [models[k] for k in names]
    p, g, pfo_h, pfo_l = [models[k] for k in ['p', 'g', 'pfo_h', 'pfo_l']]
    consumer_surplus = models.get('consumer_surplus', 0)
    externality = models.get('externality', 0)
    mix = models['incentive_mix']

    company_payoffs_safe_hq = payoffs['4-1-1']["P1"] + payoffs['4-1-1']["P2"]
    company_payoffs_safe_lq = payoffs['5-1-1']["P1"] + payoffs['5-1-1']["P2"]
    company_payoffs_unsafe_hq = payoffs['4-2-2']["P1"] + payoffs['4-2-2']["P2"]
    company_payoffs_unsafe_lq = payoffs['5-2-2']["P1"] + payoffs['5-2-2']["P2"]
    company_payoffs_vetted_hq = payoffs['4-3-3']["P1"] + payoffs['4-3-3']["P2"]
    company_payoffs_vetted_lq = payoffs['5-3-3']["P1"] + payoffs['5-3-3']["P2"]

    welfare_safe_hq = (company_payoffs_safe_hq * (1 + consumer_surplus)
                       - g * mix)
    welfare_unsafe_hq = (company_payoffs_unsafe_hq * (1 + consumer_surplus)
                         - (1-p) * (1 - pfo_h**2) * externality
                         - g * (mix * pfo_h**2 * + (1 - mix) * pfo_h))
    welfare_vetted_hq = (company_payoffs_vetted_hq * (1 + consumer_surplus)
                         - g * mix)
    welfare_unsafe_lq = (company_payoffs_unsafe_lq * (1 + consumer_surplus)
                         - (1-p) * (1 - pfo_l**2) * externality
                         - g * (mix * pfo_l**2 * + (1 - mix) * pfo_l))
    welfare_safe_lq = (company_payoffs_safe_lq * (1 + consumer_surplus)
                       - g * mix)
    welfare_vetted_lq = (company_payoffs_vetted_lq * (1 + consumer_surplus)
                         - (1-p) * (1 - pfo_l**2) * externality
                         - g * (mix * pfo_l**2 * + (1 - mix) * pfo_l))
    welfares = [welfare_safe_hq,
                welfare_unsafe_hq,
                welfare_vetted_hq,
                welfare_safe_lq,
                welfare_unsafe_lq,
                welfare_vetted_lq]
    game_welfare = np.sum([welfare * state_frequency
                           for welfare, state_frequency in zip(welfares,
                                                               ergodic.T)],
                          axis=0)
    return {**models, "game_welfare": game_welfare}
      
@method(build_payoffs, "regulatory_markets_v2_reward_mixed")
def build_payoffs(models):
    """Regulatory market payoffs when there is only a reward after catching
    unsafe firms."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    mix = models.get('incentive_mix', 0)
    
    k = models.get('decisiveness', 100)
    phi_h = models.get('phi_h', (1 - pfo_h))
    phi_l = models.get('phi_l', (1 - pfo_l))
    caught_loses_h = ((s * phi_h)**k + 1)**(-1)
    caught_loses_l = ((s * phi_l)**k + 1)**(-1)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = ((1 - pfo_h) * b / (s+1) * risk_shared
             + pfo_h * caught_loses_h * (b + B / W)
             - c)
    Π_h21 = (p * (1 - pfo_h) * (s*b / (s + 1) + s * B / W)
             + pfo_h * (1 - caught_loses_h) * B / W)
    Π_h22 = p * ( 1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
    Π_h22 = (p * (1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
             + pfo_h**2 * B/(2*W))
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = ((1 - pfo_l) * b / (s+1) * risk_shared
             + pfo_l * caught_loses_l * (b + B / W)
             - c)
    Π_l21 = (p * (1 - pfo_l) * (s*b / (s + 1)  + s * B / W)
                 + pfo_l * (1 - caught_loses_l) * B / W)
    Π_l22 = p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
    Π_l22 = (p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
             + pfo_l**2 * B/(2*W))
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g * mix
    Ω_12 = r_l + g * mix
    Ω_21 = r_h + g * (pfo_h**2 * mix + pfo_h * (1 - mix)) - λ_h
    Ω_22 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    Ω_31 = r_h + g * mix
    Ω_32 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v3_reward_mixed")
def build_payoffs(models):
    """Regulatory market payoffs when there is only a reward after catching
    unsafe firms."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    mix = models.get('incentive_mix', 0)
    
    k = models.get('decisiveness', 100)
    phi_h = models.get('phi_h', 1/s)
    phi_l = models.get('phi_l', 1/s)
    caught_loses_h = ((s * phi_h)**k + 1)**(-1)
    caught_loses_l = ((s * phi_l)**k + 1)**(-1)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = ((1 - pfo_h) * b / (s+1) * risk_shared
             + pfo_h * caught_loses_h * (b + B / W)
             - c)
    Π_h21 = (p * (1 - pfo_h) * (s*b / (s + 1) + s * B / W)
                 + pfo_h * (1 - caught_loses_h) * B / W)
    Π_h22 = p * ( 1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = ((1 - pfo_l) * b / (s+1) * risk_shared
             + pfo_l * caught_loses_l * (b + B / W)
             - c)
    Π_l21 = (p * (1 - pfo_l) * (s*b / (s + 1)  + s * B / W)
                 + pfo_l * (1 - caught_loses_l) * B / W)
    Π_l22 = p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g * mix
    Ω_12 = r_l + g * mix
    Ω_21 = r_h + g * (pfo_h**2 * mix + pfo_h * (1 - mix)) - λ_h
    Ω_22 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    Ω_31 = r_h + g * mix
    Ω_32 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v4_reward_mixed")
def build_payoffs(models):
    """Regulatory market payoffs for a mix of incentives and allows a
    schedule of measures to apply to firms detected as unsafe."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    mix = models.get('incentive_mix', 0)
    
    k = models.get('decisiveness', 100)
    phi_h = models.get('phi_h', 1/s)
    phi2_h = models.get('phi2_h,', 1/s)
    phi_l = models.get('phi_l', 1/s)
    phi2_l = models.get('phi2_l', 1/s)
    caught_loses_h = ((s * phi_h)**k + 1)**(-1)
    caught_loses_l = ((s * phi_l)**k + 1)**(-1)
    both_caught_lose_h = ((s * phi2_h)**k + 1)**(-1)
    both_caught_lose_l = ((s * phi2_l)**k + 1)**(-1)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = ((1 - pfo_h) * b / (s+1) * risk_shared
             + pfo_h * caught_loses_h * (b + B / W)
             - c)
    Π_h21 = (p * (1 - pfo_h) * (s*b / (s + 1) + s * B / W)
             + pfo_h * (1 - caught_loses_h) * B / W)
    Π_h22 = p * ( 1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
    Π_h22 = (p * (1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
             + pfo_h**2 * both_caught_lose_h * B/(2*W))
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = ((1 - pfo_l) * b / (s+1) * risk_shared
             + pfo_l * caught_loses_l * (b + B / W)
             - c)
    Π_l21 = (p * (1 - pfo_l) * (s*b / (s + 1)  + s * B / W)
                 + pfo_l * (1 - caught_loses_l) * B / W)
    Π_l22 = p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
    Π_l22 = (p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
             + pfo_l**2  * both_caught_lose_l * B/(2*W))
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g * mix
    Ω_12 = r_l + g * mix
    Ω_21 = r_h + g * (pfo_h**2 * mix + pfo_h * (1 - mix)) - λ_h
    Ω_22 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    Ω_31 = r_h + g * mix
    Ω_32 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v5_reward_mixed")
def build_payoffs(models):
    """Regulatory market payoffs for a mix of incentives and allows a
    schedule of measures to apply to firms detected as unsafe. Investigating
    a potential correction to the probability of a firm winning."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    mix = models.get('incentive_mix', 0)
    
    k = models.get('decisiveness', 100)
    # Speed impact of regulators when they catch 1 or 2 safety violators
    phi_h = models.get('phi_h', 1/s)
    phi2_h = models.get('phi2_h,', 1/s)
    phi_l = models.get('phi_l', 1/s)
    phi2_l = models.get('phi2_l', 1/s)
    # Tullock contest to determine which firm wins after
    # one safety violator is caught
    caught_loses_h = ((s * phi_h)**k + 1)**(-1)
    caught_loses_l = ((s * phi_l)**k + 1)**(-1)
    # Tullock contest to determine whether any firm wins if they are both
    # safety violators who were caught by the regulator
    both_caught_fail_h = ((s * phi2_h)**k + 1)**(-1)
    both_caught_fail_l = ((s * phi2_l)**k + 1)**(-1)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = ((1 - pfo_h) * b / (s+1) * risk_shared
             + pfo_h * caught_loses_h * (b + B / W)
             - c)
    Π_h21 = (p * (1 - pfo_h) * (s*b / (s + 1) + s * B / W)
             + (pfo_h * (1 - caught_loses_h)
                * B / W))
    Π_h22 = (p * (1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
             + (pfo_h**2 * (1 - both_caught_fail_h)
                * B/(2*W)))
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = ((1 - pfo_l) * b / (s+1) * risk_shared
             + pfo_l * caught_loses_l * (b + B / W)
             - c)
    Π_l21 = (p * (1 - pfo_l) * (s*b / (s + 1)  + s * B / W)
             + (pfo_l * (1 - caught_loses_l)
                * B / W))
    Π_l22 = (p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
             + (pfo_l**2  * (1 - both_caught_fail_l)
                * B/(2*W)))
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g * mix
    Ω_12 = r_l + g * mix
    Ω_21 = r_h + g * (pfo_h**2 * mix + pfo_h * (1 - mix)) - λ_h
    Ω_22 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    Ω_31 = r_h + g * mix
    Ω_32 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

@method(build_payoffs, "regulatory_markets_v6_reward_mixed")
def build_payoffs(models):
    """Regulatory market payoffs for a mix of incentives and allows a
    schedule of measures to apply to firms detected as unsafe. The speed of
    caught safety violators is not necessarily 1 and the risk is not
    necessarily 0."""
    names1 = ['b', 'c', 's', 'p', 'B', 'W']
    names2 = ['pfo_l', 'pfo_h', 'λ', 'r_l', 'r_h', 'g']
    b, c, s, p, B, W = [models[k] for k in names1]
    pfo_l, pfo_h, λ, r_l, r_h, g = [models[k] for k in names2]
    collective_risk = models.get('collective_risk', 0)
    risk_shared = (1 - (1-p)*collective_risk)
    mix = models.get('incentive_mix', 0)
    
    k = models.get('decisiveness', 100)
    # Win impact of regulators when they catch 1 or 2 safety violators
    phi_h = models.get('phi_h', 1/s)
    phi2_h = models.get('phi2_h,', 1/s)
    phi_l = models.get('phi_l', 1/s)
    phi2_l = models.get('phi2_l', 1/s)
    # Speed impact of regulators when they catch 1 or 2 safety violators
    theta_h = models.get('theta_h', 1/s)
    theta2_h = models.get('theta2_h,', 1/s)
    theta_l = models.get('theta_l', 1/s)
    theta2_l = models.get('theta2_l,', 1/s)
    # Risk impact of regulators when they catch 1 or 2 safety violators
    gamma_h = models.get('gamma_h', phi_h)
    gamma2_h = models.get('gamma2_h', phi2_h)
    gamma_l = models.get('gamma_l', phi_l)
    gamma2_l = models.get('gamma2_l', phi2_l)
    # Tullock contest to determine which firm wins after
    # one safety violator is caught
    caught_loses_h = ((s * phi_h)**k + 1)**(-1)
    caught_loses_l = ((s * phi_l)**k + 1)**(-1)
    # Tullock contest to determine whether any firm wins if they are both
    # safety violators who were caught by the regulator
    both_caught_fail_h = ((s * phi2_h)**k + 1)**(-1)
    both_caught_fail_l = ((s * phi2_l)**k + 1)**(-1)
    risk_shared_reg2_h = (1 - (1-p)*collective_risk * gamma2_h)
    risk_shared_reg2_l = (1 - (1-p)*collective_risk * gamma2_l)
    
    Π_h11 = B / (2*W) + b/2 - c
    Π_h12 = ((1 - pfo_h) * b / (s+1) * risk_shared
             + pfo_h * caught_loses_h * (b + B / W)
             - c)
    Π_h21 = (p * (1 - pfo_h) * (s*b / (s + 1) + s * B / W)
             + ((1 - (1 - p) * gamma_h)
                * pfo_h * (1 - caught_loses_h)
                * theta_h * s
                * B / W))
    Π_h22 = (p * (1 - pfo_h**2) * (b/2 + s*B/(2*W)) * risk_shared
             + ((1 - (1 - p) * gamma2_h) * risk_shared_reg2_h
                * pfo_h**2 * (1 - both_caught_fail_h)
                * theta2_h * s
                * B/(2*W)))
    
    Π_l11 = B / (2*W) + b/2 - c
    Π_l12 = ((1 - pfo_l) * b / (s+1) * risk_shared
             + pfo_l * caught_loses_l * (b + B / W)
             - c)
    Π_l21 = (p * (1 - pfo_l) * (s*b / (s + 1)  + s * B / W)
             + ((1 - (1 - p) * gamma_l)
                * pfo_l * (1 - caught_loses_l)
                * theta_l* s
                * B / W))
    Π_l22 = (p * ( 1 - pfo_l**2) * (b/2 + s*B/(2*W)) * risk_shared
             + ((1 - (1 - p) * gamma2_l) * risk_shared_reg2_l
                * pfo_l**2  * (1 - both_caught_fail_l)
                * theta2_l * s
                * B/(2*W)))
    
    λ_h = λ * (1 - p) * (1 - pfo_h)
    λ_l = λ * (1 - p) * (1 - pfo_l)
    
    Ω_11 = r_h + g * mix
    Ω_12 = r_l + g * mix
    Ω_21 = r_h + g * (pfo_h**2 * mix + pfo_h * (1 - mix)) - λ_h
    Ω_22 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    Ω_31 = r_h + g * mix
    Ω_32 = r_l + g * (pfo_l**2 * mix + pfo_l * (1 - mix)) - λ_l
    
    payoffs = {}
    payoffs["4-1-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-1-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-1-3"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-2-1"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-2-2"] = {"P3": Ω_21,
                        "P2": Π_h22,
                        "P1": Π_h22}
    payoffs["4-2-3"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h21,
                        "P1": Π_h12}
    payoffs["4-3-1"] = {"P3": Ω_11,
                        "P2": Π_h11,
                        "P1": Π_h11}
    payoffs["4-3-2"] = {"P3": (Ω_11 + Ω_21) / 2,
                        "P2": Π_h12,
                        "P1": Π_h21}
    payoffs["4-3-3"] = {"P3": Ω_31,
                        "P2": Π_h11,
                        "P1": Π_h11}
    
    payoffs["5-1-1"] = {"P3": Ω_12,
                        "P2": Π_l11,
                        "P1": Π_l11}
    payoffs["5-1-2"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-1-3"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l12,
                        "P1": Π_l21}
    payoffs["5-2-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-2-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-2-3"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-1"] = {"P3": (Ω_12 + Ω_22) / 2,
                        "P2": Π_l21,
                        "P1": Π_l12}
    payoffs["5-3-2"] = {"P3": Ω_22,
                        "P2": Π_l22,
                        "P1": Π_l22}
    payoffs["5-3-3"] = {"P3": Ω_32,
                        "P2": Π_l22,
                        "P1": Π_l22}

    return {**models, "payoffs": payoffs}

