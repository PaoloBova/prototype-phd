from .utils import *
from .types import *

import collections
import functools
import itertools

import more_itertools
import numpy as np

def model_builder(saved_args:dict, # a dictionary containing parameters we want to vary or hold fixed
                  exclude_args:list[str]=[], # a list of arguments that should be returned as they are
                  override:bool=False, # whether to build the grid if it is very large
                  drop_args:list[str]=['override', 'exclude_args', 'drop_args'], # a list of arguments to drop from the final result
                 ) -> dict: # A dictionary containing items for the desired models
    """Build models for all combinations of the valules in `saved_args`."""
    axes_args1 = {k: np.array(v)
                  for k,v in saved_args.items()
                  if (isinstance(v, np.ndarray)
                      or (isinstance(v, list)
                          and all(isinstance(el, (float, int)) for el in v)))}
    axes_args2 = {k: np.arange(v["start"], v["stop"], v["step"])
                  for k,v in saved_args.items()
                  if (isinstance(v, dict)
                      and v.keys() == {"start", "stop", "step"})}
    axes_args = {**axes_args1, **axes_args2}
    grid = build_grid_from_axes(list(axes_args.values()),
                                override=override)
    models = {}
    # add grid parameters first
    col_end = dict(zip(axes_args.keys(),
                            np.cumsum([v.shape[-1] if np.ndim(v)==2 else 1
                             for v in axes_args.values()])))
    col_start = {arg: col_end[arg] - (v.shape[-1] if np.ndim(v)==2 else 1)
                  for arg, v in axes_args.items()}
    for i, arg in enumerate(axes_args.keys()):
        if col_start[arg] + 1 == col_end[arg]:
            models[arg] = grid[:, col_start[arg]]
        else:
            models[arg] = grid[:, col_start[arg]:col_end[arg]]
    # add fixed parameters next
    for arg, v in saved_args.items():
        if arg not in (exclude_args
                       + drop_args
                       + list(axes_args.keys())):
            models[arg] = np.array([v for _ in grid])
    # add extra variables
    for arg in exclude_args:
        if (arg in saved_args.keys()
            and arg not in drop_args):
            models[arg] = saved_args[arg]
    return models

def find_unique_allocations(n,  # The number of items to allocate
                            k):  # The number of bins that items can be allocated to
    """Find all combinations with replacement of 'n' of the first `k` integers
    where the sum of the integers is `k`."""
    divider_locations = itertools.combinations(range(n+k-1), k-1)
    allocations = [np.diff(d, append=n+k-1, prepend=-1) - 1
                   for d in divider_locations]
    return allocations

def find_unique_counts(count_groups, current_sub_group):
    """Build a list of all unique count groups given an allocation and a list
    of existing unique counts."""
    new_count_groups = []
    combined_counts = collections.defaultdict(list)
    # First, we turn our new allocation rules into an array of all possible
    # allocations for this sub-group.
    # Each player in a sub-group follows the same allocation rules so they can
    # each have any relevant strategy.
    # Each allocation must have all relevant strategy counts recorded, including
    # those which are 0.
    categories, n_draws = current_sub_group
    n_categories = len(categories)
    counts = find_unique_allocations(n_draws, n_categories)
    n_all_categories = len(count_groups[0][0])
    allocations = np.zeros((len(counts), n_all_categories))
    for category, count in zip(categories, np.array(counts).T):
        allocations[:, int(category)] = count
    for group in count_groups:
        for allocation in allocations:
            count_seq = [*group, [*map(int, allocation)]]
            combined_count = np.sum(count_seq, axis=0)
            count_key = ":".join(map(str, map(int, combined_count)))
            combined_counts[count_key].append(count_seq)
    for repeated_counts in combined_counts.values():
        if len(repeated_counts) == 1:
            new_count_groups.append(repeated_counts[0])
        else:
            n_sub_groups = len(repeated_counts[0])
            # Convert the repeated counts into a 2D array
            x = np.vstack([np.hstack(seq) for seq in repeated_counts])
            # Lexicographic sort means that lower strategies most common on bottom
            # row (for earlier sub-groups).
            candidate = x[np.lexsort(x.T[::-1])[-1]]
            # Split row back into a list of counts for each sub-group
            # Throw an error if for some reason the sub-groups have differently
            # sized arrays (only possible if wrong values passed to this function).
            candidate = np.split(candidate, n_sub_groups)
            new_count_groups.append(candidate)
    return new_count_groups

@multi
def create_profiles(models):
    """Create strategy profiles relevant to the strategies, players, and sectors
    in models according to the given rule"""
    return models.get('profiles_rule')

@method(create_profiles, "anonymous")
def create_profiles(models):
    """Create strategy profiles for each combination of strategies. As
    the payoffs do not care who each player is (i.e. they are anonymous), we
    only create one profile for each unique strategy count. In this case we
    always keep alike strategies together with the lowest strategy on the right,
    weakly increasing as we read left, where possible. 

    When some players only
    draw from a subset of sectors, we follow a similar process for each unique
    subset of sectors that groups of players are allowed to sample from.

    Follow this routine: given one subset and the previous result of this
    routine, find all possible strategy counts using the previous set of
    strategy counts and the allowed player allocations for this subset. When
    we repeat a strategy count, keep only the strategy count (for the subset)
    associated with the previous strategy count with the highest value of
    lower strategies. This way, we have a reducer function that returns a
    list of lists of strategy count components which always add up to a unique
    strategy count.
    """

    sector_strategies = models.get('sector_strategies', {})
    allowed_sectors = models.get('allowed_sectors', {})
    n_strategies = [len(v) for v in sector_strategies.values()]
    zero_strategy = False
    for strategies in sector_strategies.values():
        if "0" in strategies:
            zero_strategy = True
    n_strategies_total = np.sum(n_strategies) + (1 - zero_strategy)

    rules = collections.defaultdict(list)
    for player, sectors in allowed_sectors.items():
        available_strategies = []
        for sector in sectors:
            available_strategies += sector_strategies[sector]
        available_strategies = np.unique(available_strategies)
        rule = ":".join(available_strategies)
        rules[rule].append(player)

    subgroups = [[str.split(rule, ':'), len(players)]
                 for rule, players in rules.items()]
    zero_counts = [[np.zeros((n_strategies_total), dtype=int)]]
    unique_counts = functools.reduce(find_unique_counts,
                                     subgroups,
                                     zero_counts)
    profiles = []
    for unique_count in unique_counts:
        player_assignments = {}
        unique_count = unique_count[1:]
        for i, subgroup_count in enumerate(unique_count):
            subgroup_key = ":".join(map(str, subgroups[i][0]))
            player_strings = rules[subgroup_key]
            player_numbers = [int("".join(string[1:]))
                              for string in player_strings]
            subgroup_players = [f"P{x}" for x in sorted(player_numbers)]
            count = subgroup_count.copy()
            for player in subgroup_players:
                strategy = np.argmax(np.array(count) > 0)
                player_assignments[player] = strategy
                count[strategy] -= 1
        profile = [player_assignments[f"P{i+1}"]
                   for i, _ in enumerate(player_assignments)]
        profile = "-".join(map(str, profile[::-1]))
        profiles.append(profile)
    return {**models, "profiles": profiles}

@method(create_profiles, "from_strategy_count")
def create_profiles(models):
    """Create all valid profiles given allowed_actions, where the count of each
    strategy is equal to the given strategy_count."""
    strategy_count = models['strategy_count']
    allowed_actions = models.get('allowed_actions', {}) # Optional
    # If provided, allowed_actions must provide rules for every player.
    assert (len(allowed_actions) == sum(strategy_count.values())
            or allowed_actions == {})
    # Create a set based on the strategy_count
    strategy_pool = [[strategy]*count for strategy, count in strategy_count.items()]
    strategy_pool = [val for sublist in strategy_pool for val in sublist]
    profiles = []
    if allowed_actions == {}:
        for profile in more_itertools.distinct_permutations(strategy_pool,
                                                            r=len(strategy_pool)):
            profile = "-".join(map(str, profile))
            profiles.append(profile)
    else:
        for profile in more_itertools.distinct_permutations(strategy_pool,
                                                            r=len(strategy_pool)):
            profile_valid = True
            for player, strategy in enumerate(string_to_tuple(profile)):
                if strategy not in allowed_actions[f"P{player + 1}"]:
                    profile_valid = False
            if profile_valid:
                profiles.append(profile)
    return {**models, "profiles": profiles}

@method(create_profiles)
def create_profiles(models):
    """Create all strategy profiles for the set of models."""
    sector_strategies = models.get('sector_strategies', {})
    allowed_sectors = models.get('allowed_sectors', {})
    n_players = models.get('n_players', len(allowed_sectors))
    n_strategies = models.get('n_strategies',
                              [len(v) for v in sector_strategies.values()])
    n_strategies_total = np.sum(n_strategies)
    if sector_strategies=={}:
        strategies = np.arange(n_strategies_total)
    else:
        strategies = np.unique([strategy
                                for sector in sector_strategies.keys()
                                for strategy in sector_strategies[sector]])
    n_profiles = n_strategies_total ** n_players
    strategy_axis = strategies[:, None]
    grid = build_grid_from_axes([strategy_axis for _ in range(n_players)])
    profiles = []
    for row in grid:
        profile = "-".join(map(str, row))
        profiles.append(profile)
    assert len(profiles) == n_profiles
    return {**models, "profiles": profiles}

@multi
def profile_filter(models):
    "Filter strategy profiles to those which satisfy the given rule."
    return models.get('profile_filter_rule')

@method(profile_filter, 'allowed_sectors')
def profile_filter(models):
    """Filter strategy profiles to only those where players are from their
    allowed sectors."""
    profiles = models.get('profiles_filtered',
                          models.get('profiles',
                                     create_profiles(models)['profiles']))
    allowed_sectors = models['allowed_sectors']
    sector_strategies = models['sector_strategies']
    profiles_filtered = []
    for k in profiles:
        k_tuple = list(map(int, k.split("-")))
        valid = True
        for i, ind in enumerate(k_tuple[::-1]):
            allowed_inds = np.hstack([sector_strategies[j]
                                      for j in allowed_sectors[f"P{i+1}"]])
            if (ind not in allowed_inds) and (str(ind) not in allowed_inds):
                valid = False
        if valid==True:
            profiles_filtered.append(k)
    return {**models, "profiles_filtered": profiles_filtered}

@method(profile_filter)
def profile_filter(models):
    """The default filter method leaves models unchanged."""
    print("""`profile_filter` called but `models` did not specify a
           `profile_filter_rule`. Try specifying one.""")
    return models

@method(profile_filter, 'relevant_to_transition')
def profile_filter(models):
    """Filter for strategy profiles relevant to the given transition."""
    ind1, ind2 = models.get('transition_indices', [None, None])
    if (ind1==None) and (ind2==None):
        return models
    sector_strategies = models['sector_strategies']
    profiles = models.get('profiles_filtered',
                          models.get('profiles',
                                     create_profiles(models)['profiles']))
    strategies1 = list(map(int, ind1.split("-")))
    strategies2 = list(map(int, ind2.split("-")))
    differ = [i1!=i2 for i1, i2 in zip(strategies1, strategies2)]
    # Check states only differ for one sector
    valid = sum(differ) == 1
    # Check that states use valid stratgy codes for each sector
    # Unlike when the states differ by more than one sector, this will only
    # happen if the transition_indices and sctor_strategies are inconsistent,
    # so we raise a value error.
    for i, sector in enumerate(sorted(sector_strategies)):
        if ((strategies1[-(i+1)] not in sector_strategies[sector]
             and str(strategies1[-(i+1)]) not in sector_strategies[sector])
            or (strategies2[-(i+1)] not in sector_strategies[sector]
                and str(strategies2[-(i+1)]) not in sector_strategies[sector])):
            valid = False
            raise ValueError("States use invalid strategy codes for some sectors.")
    if valid:
        strategies_valid = np.unique(np.hstack([strategies1, strategies2]))
        profiles_filtered = []
        for profile in profiles:
            relevant = True
            for strategy in list(map(int, profile.split("-"))):
                if strategy not in strategies_valid:
                    relevant = False
            if relevant == True:
                profiles_filtered.append(profile)
        return {**models, "profiles_filtered": profiles_filtered}
    return models

def apply_profile_filters(models):
    "Apply all profile filters listed in `profile_filters` in `models`."
    for rule in models.get('profile_filters', 
                           ["allowed_sectors",
                            "relevant_to_transition"]):
        models = profile_filter({**models, "profile_filter_rule": rule})
    return models

@method(create_profiles, "allowed_sectors")
def create_profiles(models):
    """Create all strategy profiles for the set of models."""
    profiles = thread_macro(create_profiles({**models, "profiles_rule": None}),
                            (assoc, "profile_filter_rule", "allowed_sectors"),
                            profile_filter,
                            (get, "profiles_filtered"),
                            )
    return {**models, "profiles": profiles}

