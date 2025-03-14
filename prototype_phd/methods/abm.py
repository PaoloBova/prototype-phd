import collections
import logging
import numpy as np
import pandas as pd
import random
import prototype_phd.utils as utils
import prototype_phd.data_utils as data_utils
import tqdm
from typing import Dict, List, Union, Optional, Callable, Any

def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility in simulations.

    Parameters:
    seed (int): The seed value to be used for the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    # If other libraries are used, their seeds should be set here as well.
    # Note that Autogen only offers a seed cache which must be set
    # each time a new API call to an LLM is made.

def init_agents(params, group_by_category=False, randomize=True):
    """Initialize agents by creating instances of the agent class.
    
    Parameters:
    params: A dictionary containing the parameters for the simulations.
    group_by_category: Whether to group agents by category.
    randomize: Whether to randomize the order of agent specs.
    
    Returns:
    agents: A list of agents or a dictionary of agents grouped by category.
    
    Notes:
    - The `agent_specs` parameter should be a list of dictionaries.
    - The `agent_secrets` parameter should be a dictionary of secrets.
    - For each spec:
        - The `agent_category` parameter is optional (default is None).
        - The `num_agents` parameter is optional (default is 1).
        - The `agent_spec_id` parameter is optional (default is None).
    """
    agent_specs = params.get('agent_specs', [])
    agent_secrets = params.get('agent_secrets', {})
    agents = {} if group_by_category else []
    if randomize:
        # Randomize order of agent specs
        random.shuffle(agent_specs)
    i == 0
    for spec in agent_specs:
        agent_class = spec['agent_class']
        num_agents = spec.get('num_agents', 1)
        agent_params = spec.get('agent_params', {})
        spec_id = spec.get('agent_spec_id', None)
        secrets = agent_secrets.get(spec_id, {})
        agent_category = spec.get('agent_category', None)
        for _ in range(num_agents):
            # Assign each agent a unique id based on insertion order
            i += 1
            agent_id = i
            agent = agent_class(agent_id=agent_id, **agent_params, **secrets)
            if group_by_category:
                agents.setdefault(agent_category, []).append(agent)
            else:
                agents.append(agent)
    return agents

def run(model, parameters):
    logging.info(f"Starting simulation with {len(model.agents)} agents.")
    
    model.collect_stats(parameters)
    for _ in range(model.num_rounds):
        logging.info(f"Round {model.tick} begins.")
        model.step(parameters)
        #TODO: Allow more flexibility in when to collect what.
        model.collect_stats(parameters)
        logging.info(f"Round {model.tick} ends.")
        logging.info("-" * 50)

    return model.agent_results, model.model_results
    
def run_simulation(params):
    """Initialize the agents and model, then run the model."""
    model_class = params['model_class']
    agents = init_agents(params)
    model = model_class(agents, params)
    agent_results, model_results = run(model, params)
    return model, agent_results, model_results

def run_multiple_simulations(params:Dict,
                             secrets:Dict={},
                             custom_collect_fn: Optional[Callable[[dict], dict]] = None) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations.
    secrets: A dictionary of secrets to be used in the simulations.
    custom_collect_fn: A custom function to collect additional results.
        Returns a dict of: result_type_string -> value
    
    Returns:
    A dictionary of DataFrames and objects which are JSON serializable.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    # If params is a dictionary, convert it into a list of dictionaries
    params_list = utils.dict_list(params) if isinstance(params, dict) else params
    # Assert that params is a list of dictionaries
    assert all(isinstance(params, dict) for params in params_list)
    logging.info(f"Number of simulations: {len(params_list)}")

    agent_results_all = []
    model_results_all = []
    # Create a nested (two-layer) default dictionary for storing results of any type.
    # This dictionary maps keys (e.g., result types) to inner dictionaries
    # which map simulation_run_ids to values.
    custom_data: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    # All params in params_list should have the same `simulation_id`
    # Only their `simulation_run` and `simulation_run_id` should differ
    simulation_id =  params_list[0]['simulation_id']
    assert params_list[0].get('simulation_id') is not None
    assert all(params['simulation_id'] == simulation_id
               for params in params_list)

    # We need to create the random ids for each simulation run before
    # we set the random seeds for the simulation runs.
    params_list = [{**params, "simulation_run_id": data_utils.create_id(verbose=False)}
                   for params in params_list]
    for i, params in tqdm.tqdm(enumerate(params_list)):
        set_random_seed(params['seed'])
        # We keep secrets separate from the rest of the params as we don't want
        # to expose them in the results
        args = {**params, **secrets}
        model, agent_results, model_results = run_simulation(args)
        # Add columns to identify the simulation id, run, and run id
        simulation_run_id = params["simulation_run_id"]
        for res in agent_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        for res in model_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = i + 1
            res['simulation_run_id'] = simulation_run_id
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
        
        if custom_collect_fn:
            custom_result = custom_collect_fn({**args, "model": model})
            for k, v in custom_result.items():
                custom_data[k][simulation_run_id] = v
    
    # Create DataFrames from the results
    agent_df = pd.DataFrame(agent_results_all)
    model_df = pd.DataFrame(model_results_all)
    
    # Return a dictionary of DataFrames and objects which are JSON serializable
    data = {'agent': agent_df,
            'model': model_df,
            **custom_data,
            "params": [data_utils.filter_dict_for_json(params)
                       for params in params_list]}
    return data

def sanitize_params(params:Union[Dict, List[Dict]]) -> List[Dict]:
    """Sanitize the parameters for the simulations and convert to list."""
    # If params is a dictionary, convert it into a list containing a dictionary
    params_list = [params] if isinstance(params, dict) else params
    # Assert that params is a list of dictionaries
    assert all(isinstance(params, dict) for params in params_list)
    logging.info(f"Number of simulations: {len(params_list)}")

    # All params in params_list should have the same `simulation_id`
    # Only their `simulation_run` and `simulation_run_id` should differ
    simulation_id =  params_list[0]['simulation_id']
    assert params_list[0].get('simulation_id') is not None
    assert all(params['simulation_id'] == simulation_id
               for params in params_list)
    
    # We need to create the random ids for each simulation run before
    # we set the random seeds for the simulation runs.
    run_ids = data_utils.create_ids(len(params_list))
    i = 0
    for params, run_id in zip(params_list, run_ids):
        i += 1
        utils.set_nested_value(params, ['simulation_run_id'], run_id)
        utils.set_nested_value(params, ['simulation_run'], i + 1)
    return params_list

def sanitize_filepaths(filepaths, simulation_id):
    """Sanitize the filepaths for saving the results."""
    # Ensuer we have a filepath for saving the inputs
    if not 'inputs' in filepaths:
        filepaths['inputs'] = f"{simulation_id}"
    # Make sure we at least have filepaths for saving the agent and model results
    if not 'model_results' in filepaths:
        filepaths['model_results'] = "model_results"
    if not 'agent_results' in filepaths:
        filepaths['agent_results'] = "agent_results"
    return filepaths

def run_sims_online(params:Union[Dict, List[Dict]],
                    secrets:Dict={},
                    custom_collect_fn: Optional[Callable[[dict], dict]] = None,
                    collect_as_vectors:bool=False) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations. Specify as a list of
        dictionaries or a single dictionary.
    secrets: A dictionary of secrets to be used in the simulations.
        We keep secrets separate from the rest of the params as we don't want
        to expose them in the results
    custom_collect_fn: A custom function to collect additional results.
        Returns a dict of: result_type_string -> value
    collect_as_vectors: Whether the collected results are dicts of vectors or
      scalars. In the former case, we need to concatenate the vectors before we
      can construct a dataframe from the results.
    
    Returns:
    A dictionary of DataFrames and objects which are JSON serializable.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    params_list = sanitize_params(params)
    agent_results_all = []
    model_results_all = []
    # Create a nested (two-layer) default dictionary for storing results of any type.
    # This dictionary maps keys (e.g., result types) to inner dictionaries
    # which map simulation_run_ids to values.
    custom_data: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)
    simulation_id =  params_list[0]['simulation_id']
    for params in params_list:
        set_random_seed(params['seed'])
        args = {**params, **secrets}
        model, agent_results, model_results = run_simulation(args)
        # Add columns to identify the simulation id, run, and run id
        simulation_run_id = params['simulation_run_id']
        simulation_run = params["simulation_run"]
        for res in agent_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = simulation_run
            res['simulation_run_id'] = simulation_run_id
        for res in model_results:
            res['simulation_id'] = simulation_id
            res['simulation_run'] = simulation_run
            res['simulation_run_id'] = simulation_run_id
        agent_results_all.extend(agent_results)
        model_results_all.extend(model_results)
        if custom_collect_fn:
            custom_result = custom_collect_fn({**args, "model": model})
            for k, v in custom_result.items():
                custom_data[k][simulation_run_id] = v

    if collect_as_vectors:
        agent_results_new = {}
        if agent_results_all:
            for k in agent_results_all[0].keys():
                agent_results_new[k] = np.hstack([d[k] for d in agent_results_all])
        agent_results_new = data_utils.sanitize_dict_values(agent_results_new)
        
        model_results_new = {}
        if model_results_all:
            for k in model_results_all[0].keys():
                model_results_new[k] = np.hstack([d[k] for d in model_results_all])
        model_results_new = data_utils.sanitize_dict_values(model_results_new)
    else:
        agent_results_new = agent_results_all
        model_results_new = model_results_all

    # Create DataFrames from the results
    agent_df = pd.DataFrame(agent_results_new)
    model_df = pd.DataFrame(model_results_new)
    
    # Return a dictionary of DataFrames and objects which are JSON serializable
    data = {'agent': agent_df,
            'model': model_df,
            **custom_data,
            "params": [data_utils.filter_dict_for_json(params)
                       for params in params_list]}
    return data

def run_sims_offline(params:Union[Dict, List[Dict]],
                     secrets:Dict,
                     filepaths:Dict) -> Dict:
    """Run multiple simulations, assuming the model saves results to disk.
    
    Parameters:
    params: The parameters for the simulations. Specify as a list of
        dictionaries or a single dictionary.
    secrets: A dictionary of secrets to be used in the simulations.
        We keep secrets separate from the rest of the params as we don't want
        to expose them in the results
    filepaths: A dictionary of filepaths to save data to.
    
    Returns:
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    params_list = sanitize_params(params)
    simulation_id = params_list[0]['simulation_id']
    filepaths = sanitize_filepaths(filepaths, simulation_id)
    data_utils.save_inputs(filepaths['inputs'], params_list)
    for params in params_list:
        set_random_seed(params['seed'])
        args = {**params, **secrets, "filepaths": filepaths}
        run_simulation(args)
    return filepaths

def paramscan(params:Union[Dict, List[Dict]],
              secrets:Dict={},
              filepaths:Dict={},
              collect_as_vectors:bool=False) -> Dict:
    """Run multiple simulations and collect the results.
    
    Parameters:
    params: The parameters for the simulations. Specify as a list of
        dictionaries or a single dictionary.
    secrets: A dictionary of secrets to be used in the simulations.
    filepaths: A dictionary of filepaths to save data to.
    collect_as_vectors: Whether the collected results are dicts of vectors or
      scalars. In the former case, we need to concatenate the vectors before we
      can construct a dataframe from the results.
    
    Returns:
    Either
      - A dictionary of DataFrames and objects which are JSON serializable.
      - A dictionary of filepaths where the results are saved.
    
    See Agents.jl `Agents.paramscan` method for a similar API.
    """
    if not filepaths:
        data = run_sims_online(params, secrets, collect_as_vectors)
        return data
    else:
        # If filepaths is nonempty, assume results data is
        # automatically saved to disk as the model runs.
        filepaths = run_sims_offline(params, secrets, filepaths)
        return filepaths
