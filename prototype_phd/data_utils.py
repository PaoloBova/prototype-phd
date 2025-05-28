import autogen
import collections
import datetime
import hashlib
import json
import logging
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import networkx
import os
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
import pprint
import random
import regex
from scipy.stats import qmc
import subprocess
import sys
import tqdm
from typing import Any, Dict, List, Union, Tuple
import uuid
import yaml

def setup_logging(log_path='logs/chat_logs.log', level=logging.INFO):
    # Get the directory portion of the log_path.
    log_dir = os.path.dirname(log_path)
    # Create the directory (and any intermediate directories) if it doesn't exist.
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_path, level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def is_plain_word(word):
    return word.isalpha()

def generate_random_phrase(words, num_words=3):
    return '_'.join(random.sample(words, num_words))

def create_id(path_to_data='/usr/share/dict/words', verbose=True):
    """Create a unique identifier based on a random phrase and a UUID.
    
    Parameters:
    - path_to_data: The path to a file containing a list of words.
    - verbose: Whether to print the random phrase and sim ID.
    
    Returns:
    - A unique identifier string.
    
    Note:
    - If the file at `path_to_data` does not exist, a UUID will be used instead.
    - You may wish to find a list of words to use as the dictionary file. On
    Unix systems, you can use `/usr/share/dict/words`. See the following link
    for more options: https://stackoverflow.com/questions/18834636/random-word-generator-python
    """
    try:
        with open(path_to_data, 'r') as f:
            words = [line.strip() for line in f if is_plain_word(line.strip())]
        sim_id = generate_random_phrase(words)
        if verbose:
            print(f"Random Phrase: {sim_id}")
        sim_id = f"{sim_id}_{str(uuid.uuid4())[:8]}"
        if verbose:
            print(f"Sim ID: {sim_id}")
        return sim_id
    except Exception as e:
        if verbose:
            print(f"You got exception {e}. Defaulting to a UUID.")
        sim_id = str(uuid.uuid4())
        if verbose:
            print(f"Sim ID: {sim_id}")
        return sim_id

def create_ids(num_ids):
    """
    Generate a given number of unique identifiers (UUIDs).

    Parameters:
    - num_ids: The number of unique identifiers to generate.

    Returns:
    - A list of unique identifier strings.
    
    Notes:
    - Use this function to efficiently generate multiple unique identifiers
      at once (and when you don't care about the ids being human-readable).
    """
    return [str(uuid.uuid4()) for _ in range(num_ids)]

def get_current_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        commit = "git command failed, are you in a git repository?"
    return commit

def display_chat_messages(filepath):
    # Load the chat histories from the JSON file
    with open(filepath, 'r') as f:
        chat_histories = json.load(f)

    # Iterate over the chat histories
    for chat_history in chat_histories:
        # Print the chat id
        for chat_id, chats in chat_history.items():
            print(f'## Chat ID: {chat_id}\n')
            # Iterate over the chats in the chat history
            for chat in chats:
                # Pretty print the chat
                pprint.pprint(chat)

def save_name(params,
              hash_filename=False,
              sensitive_keys=[], 
              max_depth=1,
              max_length=100):
    """Create a filename stub from a dictionary of parameters."""
    # Convert the dictionary to a list of strings
    param_list = []
    for k, v in sorted(params.items()):  # Sort items by key
        if k in sensitive_keys:
            continue

        if isinstance(k, str):
            if k == '':
                raise TypeError(f"Unsupported key type: empty string")
        elif isinstance(k, (int)):
            k = str(k)
        else:
            raise TypeError(f"Unsupported key type for key {k}: {type(k)}")

        if isinstance(v, (float, int)):
            # Round the float to 2 decimal places and replace the decimal point with 'p'
            v = str(round(v, 2)).replace('.', 'p').replace('-', 'minus')
        elif isinstance(v, bool):
            v = 1 if v else 0
        elif isinstance(v, (str, type(None))):
            v = str(v)
            if v == '':
                v = 'empty_string'
        elif callable(v):
            if hasattr(v, '__self__') and v.__self__ is not None:
                v = f"{v.__self__.__class__.__name__}.{v.__name__}"
            else:
                v = v.__name__
        elif isinstance(v, complex):
            v = str(v).replace('+', 'add').replace('-', 'minus')
        elif isinstance(v, datetime.datetime):
            v = v.isoformat()
        elif isinstance(v, uuid.UUID):
            v = str(v)
        elif isinstance(v, dict) and max_depth > 0:
            nested_stub = save_name(v,
                                    hash_filename,
                                    sensitive_keys,
                                    max_depth - 1,
                                    max_length // 2)
            v = f"begin_dict_{nested_stub}_end_dict"
        else:
            continue  # Skip unsupported types
        param_list.append(f"{k}_{v}")
    
    # Join the list into a single string with underscores
    param_str = "__".join(param_list)
    
    # Ensure filename is not too long
    if len(param_str) > max_length:
        param_str = param_str[:max_length]
    param_str = param_str[:255] # Bash has a hard limit of 255 characters for byte strings
    
    if len(param_str) == 0:
        raise ValueError("No parameters to save")
    if len(param_str) > 255:
        print(ValueError("Filename too long for bash. Consider excluding some variables if truncation is unacceptable."))
    
    # Hash the string to ensure uniqueness if hash_filename is True
    filename_stub = hashlib.md5(param_str.encode()).hexdigest() if hash_filename else param_str
    
    return filename_stub

def serialize_graphs(graphs):
    """Serialize a dictionary of NetworkX graphs to a JSON-serializable format."""
    serialized_graphs = {}
    for key, graph in graphs.items():
        # Convert node labels to integers
        G = networkx.convert_node_labels_to_integers(graph)

        # Convert attributes (e.g. block) to native Python int
        for node, data in G.nodes(data=True):
          if 'block' in data:
            data['block'] = int(data['block'])
        if 'partition' in G.graph:
            G.graph['partition'] = [list(s) for s in G.graph['partition']]

        serialized_graph = networkx.adjacency_data(G)
        serialized_graphs[key] = serialized_graph
    return serialized_graphs

def filter_dict_for_json(d):
    """
    Recursively filter out values from a dictionary that cannot be serialized to JSON.

    Parameters:
    d: a dictionary

    Returns:
    A new dictionary with only JSON serializable values.
    """
    filtered_dict = {}

    for key, value in d.items():
        if isinstance(value, dict):
            filtered_dict[key] = filter_dict_for_json(value)
        elif isinstance(value, list):
            filtered_list = []
            for item in value:
                if isinstance(item, dict):
                    filtered_list.append(filter_dict_for_json(item))
                else:
                    try:
                        json.dumps(item)
                        filtered_list.append(item)
                    except TypeError:
                        continue
            filtered_dict[key] = filtered_list
        else:
            try:
                json.dumps(value)
                filtered_dict[key] = value
            except TypeError:
                continue

    return filtered_dict

def append_data(data, file_path, format='csv'):
    if format == 'csv':
        file_exists = os.path.isfile(file_path)
        data.to_csv(file_path, mode='a', header=not file_exists, index=False)
    elif format == 'hdf5':
        with pd.HDFStore(file_path, mode='a') as store:
            store.append('my_key', data, format='table', data_columns=True)
    else:
        raise ValueError("Unsupported format")

def append_ndjson(record, file_path):
    with open(file_path, 'a') as f:
        f.write(json.dumps(record))
        f.write('\n')

def read_ndjson(file_path):
    results = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results

def configure_logging_console(level: int = logging.INFO):
    """
    Ensure there’s a StreamHandler on the root logger.
    Doesn’t call basicConfig so it won’t remove existing file handlers.
    """
    root = logging.getLogger()
    # Only add a console handler if none exists yet
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console.setFormatter(fmt)
        root.addHandler(console)
    root.setLevel(level)

def read_config(config_path: str) -> Dict[str, Any]:
    """Read JSON or YAML configuration file."""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

def read_data(data_path: str) -> pd.DataFrame:
    """Read data from CSV file or directory."""
    path = Path(data_path)
    if path.is_dir():
        # Read all CSV files in directory
        dfs = []
        for file_path in path.glob("*.csv"):
            df = pd.read_csv(file_path)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    else:
        # Read single CSV file
        return pd.read_csv(path)

def filter_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter data based on configuration.
    
    Args:
        df: Raw data frame
        config: Configuration dictionary
    
    Returns:
        Filtered DataFrame
    
    Example config:
    {
        "filters": {
            "model": ["gpt-3.5-turbo", "!gpt-4"],
            "task_source": ["source1", "!source2"]
        }
    }
    """
    if "filters" in config:
        for col, values in config["filters"].items():
            # support negative filters via "!"-prefix
            if isinstance(values, list):
                # separate positive and negative rules
                pos = [v for v in values if not (isinstance(v, str) and v.startswith("!"))]
                neg = [v[1:] for v in values if isinstance(v, str) and v.startswith("!")]
                if pos:
                    df = df[df[col].isin(pos)]
                if neg:
                    df = df[~df[col].isin(neg)]
            else:
                # single value filter
                if isinstance(values, str) and values.startswith("!"):
                    df = df[df[col] != values[1:]]
                else:
                    df = df[df[col] == values]
    return df

class NumpyEncoder(json.JSONEncoder): 
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert NumPy array to list
        # Let the base class default method raise the TypeError
        return super().default(obj)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

class CombinedEncoder(NumpyEncoder, DateTimeEncoder):
    pass

def save_data(data, data_dir=None, append=False):
    """
    Save each dataframe in `data` to the given folder. 
    
    Parameters:
    - data: A dictionary where the keys are the names of the files and the values are the dataframes to save.
    - data_dir: The directory to save the data in. If not provided, a random directory will be generated.
    - append: If True, append the data to the existing file. If False, overwrite the file.
    
    Usage:
    ```{python}
    data = {'df': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 'dict': {'key1': 'value1', 'key2': 'value2'}}
    save_data(data, 'my_folder')
    If no folder name is provided, generate a random one
    ```
    """
    if data_dir is None:
        data_dir = f'data/{create_id()}'
    # Create the folder if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    logging.info(f"Saving data to: {data_dir}")

    # Iterate over the items in the dictionary
    for key, value in data.items():
        csv_path = os.path.join(data_dir, f'{key}.csv')
        json_path = os.path.join(data_dir, f'{key}.json')

        # DataFrame -> CSV
        if isinstance(value, pd.DataFrame):
            if append and os.path.isfile(csv_path):
                value.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                value.to_csv(csv_path, index=False)

        # If the value is a dictionary of networkx graphs, save it as a JSON file
        elif isinstance(value, dict) and all(isinstance(graph, networkx.Graph)
                                             for graph in value.values()):
            serialized_graphs = serialize_graphs(value)
            if append and os.path.isfile(json_path):
                append_ndjson(serialized_graphs, json_path)
            else:
                with open(json_path, 'w') as f:
                    json.dump(serialized_graphs, f)
        # If the value is a dictionary or list of dictionaries, save it as a
        # JSON file
        elif isinstance(value, (dict, list)):
            if append and os.path.isfile(json_path):
                append_ndjson(value, json_path)
            else:
                with open(json_path, 'w') as f:
                    # We often need to save numpy arrays, so we use a custom
                    # encoder to handle this
                    # indent=2 makes the JSON file human-readable
                    json.dump(value, f, cls=CombinedEncoder, indent=2)

def save_chunk(filepaths, chunk, filepath_key):
    """Save a single chunk of data to an HDF5 file, appending if the file exists."""
    filepath = os.path.join(filepaths['data_dir'], 
                            filepaths.get(filepath_key, filepath_key))
    filepath = f"{filepath}.hdf5"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    chunk_df = pd.DataFrame(chunk)
    
    with h5py.File(filepath, 'a') as f:
        for column in chunk_df.columns:
            data = chunk_df[column].values
            if column in f:
                # Append data to the existing dataset
                dataset = f[column]
                dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                dataset[-data.shape[0]:] = data
            else:
                maxshape = (None,)
                f.create_dataset(column,
                                 data=data,
                                 maxshape=maxshape, chunks=True)
    
    print(f"Saved chunk to {filepath}")

def save_inputs(filename, inputs, data_dir='data'):
    """Save the inputs to filepath if no such file exists."""
    inputs_dir = os.path.join(data_dir, 'inputs')
    filepath = os.path.join(inputs_dir, f'{filename}.json')
    inputs = [filter_dict_for_json(d) for d in inputs]
    if not os.path.exists(filepath):
        save_data({filename: inputs}, data_dir=inputs_dir)

def save_sim_to_tracker(data_dir: str, sim_id: str, batch_id: Union[str, None] = None):
    """
    Save simulation metadata to the tracker file.

    Args:
        data_dir (str): The directory where the tracker file is located.
        sim_id (str): The simulation ID.
        batch_id (str, optional): The batch ID. Defaults to None.
    """
    # Define the path to the simulation tracker file
    sim_tracker_path = os.path.join(data_dir, "sim_tracker.csv")

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Check if the simulation tracker file exists
    file_exists = os.path.isfile(sim_tracker_path)

    # Create a DataFrame with the simulation metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sim_metadata = pd.DataFrame({
        "sim_id": [sim_id],
        "batch_id": [batch_id],
        "timestamp": [timestamp]
    })

    # Append the simulation metadata to the tracker file, creating the file if it doesn't exist
    try:
        if file_exists:
            sim_metadata.to_csv(sim_tracker_path, mode='a', header=False, index=False)
        else:
            print("Creating a new simulation tracker file.")
            sim_metadata.to_csv(sim_tracker_path, mode='w', header=True, index=False)
        print(f"Simulation ID {sim_id} has been added to the tracker.")
    except Exception as e:
        print(f"An error occurred while updating the simulation tracker: {e}")

def save_plots(plots, plots_dir=None):
    """
    Save each plot in `plots` to the given folder.
    
    Parameters
    ----------
    plots : dict
        A dictionary where the keys are the plot names and the values are the
        plot objects.
    data_dir : str
        The directory to save the plots in. If not provided, a random directory
        will be generated.
    
    Usage:
    ```{python}
    save_plots({'plot1': fig1, 'plot2': fig2}, data_dir='my_folder')
    ```
    """
    if plots_dir is None:
        plots_dir = f'plots/{create_id()}'
    # Create the folder if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Iterate over the items in the dictionary
    for filename_stub, plot in plots.items():
        # If plot is a matplotlib figure, save it as a PNG file
        if isinstance(plot, plt.Figure):
            filepath = os.path.join(plots_dir, f'{filename_stub}.png')
            # plt.tight_layout()
            plot.savefig(filepath, bbox_inches="tight")
            print(f"Saved file: {filepath}")
        # If plot is a plotly figure, save it as an HTML file
        elif isinstance(plot, go.Figure):
            filepath = os.path.join(plots_dir, f'{filename_stub}.html')
            pio.write_html(plot, filepath)
            print(f"Saved file: {filepath}")
        elif isinstance(plot, FuncAnimation):
            # This is a matplotlib animation. Save it as a GIF file
            filepath = os.path.join(plots_dir, f'{filename_stub}.gif')
            plot.save(filepath, writer='imagemagick')
            print(f"Saved file: {filepath}")

def extract_data(message: str, data_format: Dict[str, type]) -> List[Dict[str, Any]]:
    """Extracts data from a message according to a specified format.

    Args:
        message (str): The message to extract data from.
        data_format (Dict[str, type]): A dictionary specifying the expected keys and their corresponding types in the data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the extracted data. Each dictionary has keys and values corresponding to the data_format argument.
    """
    # Use a regular expression to extract all JSON strings from the message.
    matches = regex.findall(r'\{(?:[^{}]|(?R))*\}', message)
    data = []
    for match in matches:
        # Parse the JSON string into a dictionary.
        try:
            new_data = json.loads(match)
        except json.JSONDecodeError:
            print("Could not parse a JSON string.")
            continue

        # Validate the keys in the dictionary.
        expected_keys = set(data_format.keys())
        if set(new_data.keys()) != expected_keys:
            print("Received unexpected keys.")
            continue

        # Validate the types of the values in the dictionary.
        valid_data = True
        for key, expected_type in data_format.items():
            if not isinstance(new_data[key], expected_type):
                print(f"Received data with incorrect type for key '{key}'.")
                valid_data = False
                break

        if valid_data:
            data.append(new_data)

    return data

def sanitize_dict_values(results_dict):
    """
    Sanitizes the values in the input dictionary.

    If a value is a numpy array, it is reshaped to 1D and resized to match the
    length of the longest array in the dictionary.

    If a value is not a numpy array, it is converted into a numpy array with the
    same length as the longest array in the dictionary.

    If the input dictionary is empty or None, a warning is printed and an empty
    dictionary is returned.

    Parameters:
    results_dict (dict): The input dictionary to sanitize.

    Returns:
    dict: The sanitized dictionary.
    """

    if not results_dict:
        print("Warning: Input dictionary is empty or None.")
        return {}
    
    max_length = max(len(v) for v in results_dict.values() if isinstance(v, np.ndarray))
    
    for key, value in results_dict.items():
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                print(f"Warning: {key} is not a 1D array. Reshaping to 1D.")
                value = value.ravel()
            if len(value) < max_length:
                value = np.resize(value, max_length)
        else:
            value = np.full(max_length, value)
        results_dict[key] = value

    return results_dict

def recursive_flatten(data, parent_key='', sep='_'):
    """
    Flatten a nested dictionary by concatenating keys with underscores.

    Parameters:
      data (dict): Input nested dictionary.
      parent_key (str): The parent key for the current dictionary.
      sep (str): The separator to use between concatenated keys.

    Returns:
      dict: Flattened dictionary with concatenated keys.
    """
    items = []
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, collections.abc.Mapping):
            items.extend(recursive_flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def json_to_df(data):
    """
    Convert a nested dictionary representing JSON data into a pandas DataFrame.

    The number of rows is determined automatically:
      - If any top-level element is a list, the number of rows is set to the length
        of the longest list found.
      - If no top-level list exists, a single row is assumed.
    
    For each key-value pair from the flattened dictionary:
      - If the value is a list and its length matches the row count, it is used directly.
      - If the value is a list of different length, we assume it is a string and
        replicate it to create a column with the same string for every row.
      - For non-list values, the single value is replicated to create a column with the
        same value for every row.
    
    Nested dictionaries are flattened recursively by concatenating keys with underscores.
    
    Parameters:
      data (dict): Input nested dictionary from JSON.
    
    Returns:
      DataFrame: DataFrame constructed from the processed data.
    """
    # Determine number of rows based on top-level lists
    top_level_lists = [v for v in data.values() if isinstance(v, list)]
    num_rows = max((len(lst) for lst in top_level_lists), default=0)
    if num_rows == 0:
        num_rows = 1

    flat_data = recursive_flatten(data)

    df_dict = {}
    for key, value in flat_data.items():
        if isinstance(value, list):
            if len(value) == num_rows:
                df_dict[key] = value
            else:
                df_dict[key] = str(value)
        else:
            df_dict[key] = [value] * num_rows

    return pd.DataFrame(df_dict)

def get_autogen_chat_results(model, simulation_run_id):
    """Get the autogen chat results from the model and ensure they are in a JSON
    serializable format."""
    chat_results = {agent.name: agent.chat_messages
                    for agent in model.agents}
    
    # Chat messages are not JSON serializable, so build JSON serializable dicts
    # from them as follows:
    
    # TODO: Chat messages lack data on who sent each message. This makes it very
    # difficult to track who is who in the conversation. This should be fixed.
    
    chat_results = collections.defaultdict(list)
    for agent in model.agents:
        agent_key = str(agent.name)
        chat_messages = agent.chat_messages
        for peer_agent, chat in chat_messages.items():
            chat_id = f"agent1:{agent_key}_agent2:{str(peer_agent.name)}_sim:{simulation_run_id}"
            chat_results[chat_id].append(chat)
    return chat_results

def get_autogen_usage_summary(model):
    """Get the autogen usage summary from the model."""
    usage_summary = autogen.gather_usage_summary(model.agents)
    return usage_summary

def get_chat_data(args):
    model = args.get('model', None)
    simulation_run_id = args.get('simulation_run_id', None)
    if model:
        data = {"usage_summaries": get_autogen_usage_summary(model),
                "chat_results": get_autogen_chat_results(model, simulation_run_id)}
        return data

def get_graph_data(args):
    model = args.get('model', None)
    if model:
        return {"graphs": model.graph}

def get_llm_network_data(args):
    return {**get_chat_data(args), **get_graph_data(args)}

def setup_project(save_tracker=True,
                  data_dir_root='data',
                  plots_dir_root='plots',
                  simulation_id=None,
                  log_path='logs/default_logs.log'):
    """Set up the project by creating ids and directories.
    
    Returns:
    - simulation_id: A unique identifier for the simulation.
    - current_commit: The current git commit hash.
    - data_dir: The directory to save the simulation data in.
    - plots_dir: The directory to save the plots in.
    
    Usage:
    ```{python}
    simulation_id, current_commit, data_dir, plots_dir = setup_project()
    ```
    
    This function should be called at the beginning of a script to set up the
    project directories and logging.
    
    The simulation_id and current_commit can be used to uniquely identify the
    simulation and the version of the code used."""
    # Simulation metadata
    if simulation_id is None:
        simulation_id = create_id()
    current_commit = get_current_git_commit()

    # Directories
    data_dir = f"{data_dir_root}/{simulation_id}"
    plots_dir = f"{plots_dir_root}/{simulation_id}"

    # Save sim to tracker
    if save_tracker:
        save_sim_to_tracker(data_dir_root, simulation_id)

    # Setup logging
    setup_logging(log_path=log_path)
    return simulation_id, current_commit, data_dir, plots_dir

def get_latest_sim_id(file_path):
    """
    Get the latest simulation ID from the sim_tracker file.
    """
    df = pd.read_csv(file_path)
    if df.empty:
        return ValueError(f"File {file_path} is empty.")
    if 'sim_id' not in df.columns:
        raise ValueError(f"Column 'sim_id' not found in {file_path}")
    # Take the most recent row as per the timestamp column and return the simulation ID.
    latest_time = df['timestamp'].max()
    latest_sim_id = df.loc[df['timestamp'] == latest_time, 'sim_id'].values[-1]
    return latest_sim_id

def collect_stats_default(model, parameters): 
    for agent in model.agents:
        model.agent_results.append({
            'round': model.tick,
            'agent_id': agent.name,
            'decision': agent.state["decision"],
            'reasoning': agent.state.get("reasoning", None)
        })
    model.model_results.append({
        'round': model.tick,
        'num_agents': len(model.agents),
        **parameters
    })

def generate_qmc_samples(param_limits: Dict[str, Tuple[float, float]], n_samples: int) -> Dict[str, np.ndarray]:
    """
    Generate quasi Monte Carlo samples for a set of one-dimensional parameters.

    This function uses a Sobol sequence to produce a quasi-random sample in a multi-dimensional space.
    Each parameter is assumed to be one-dimensional and its limits are given by a (min, max) tuple.
    The generated samples for each parameter are scaled to the corresponding range.

    Args:
        param_limits (Dict[str, Tuple[float, float]]): A dictionary mapping each parameter name to a tuple (min, max)
            that defines the range of that parameter.
        n_samples (int): The number of samples to generate.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping each parameter name to a numpy array of shape (n_samples,)
            containing the quasi Monte Carlo sample values scaled to the parameter's limits.

    Example:
        >>> param_limits = {'a': (0, 10), 'b': (5, 15)}
        >>> samples = generate_qmc_samples(param_limits, 100)
        >>> samples['a']  # 100 samples between 0 and 10
        >>> samples['b']  # 100 samples between 5 and 15
    """
    dim = len(param_limits)
    sampler = qmc.Sobol(d=dim, scramble=True)
    sample = sampler.random(n=n_samples)  # shape: (n_samples, dim), values in [0, 1)
    
    sample_scaled = {}
    keys = list(param_limits.keys())
    for i, key in enumerate(keys):
        low, high = param_limits[key]
        # Scale the i-th column of sample from [0, 1) to [low, high]
        sample_scaled[key] = qmc.scale(sample[:, [i]], low, high)[:, 0]
    return sample_scaled

def bin_data_by_power(df: pd.DataFrame,
                      col: str,
                      base: float=2.0) -> pd.DataFrame:
    """
    Bin values in `col` in df by powers of `base`.
    
    
    Key points:
      1. Negative values go into a (-∞, 0) bin with no exponent.
      2. Positive values are binned into intervals of powers of `base`.
         - Always include exponents from 0 up to ceil(log_base(max_pos)).
         - If the smallest positive value is < 1, include negative exponents 
           down to floor(log_base(min_pos)).
      4. We append -∞ and +∞ to ensure all values fit into a bin, so bin_mid or
      bin_power can become ±∞ or NaN for negative/zero rows.

    Returns:
      A DataFrame copy with new columns:
        - bin (Interval)
        - bin_left, bin_right, bin_mid
        - bin_power (Int or NaN)
    """
    # force a real copy so we never write to a slice
    df = df.copy()
    assert col in df.columns
    assert base > 1
    # Step 1: determine upper and lower bounds of bin edges for positive values
    max_val = df[col].max(skipna=True)
    # Note: log_b(y) = log_k(y) / log_k(b)
    # Letting k be 2, b be base, and y be max_val, we compute log_b(y) as:
    max_val_power = int(np.ceil(np.log2(max_val) / np.log2(base)))
    # Note: we use the integer ceiling power because we want to include the
    # maximum value in the bins.
    min_val = df[col].min(skipna=True)
    min_val_power = int(np.floor(np.log2(min_val) / np.log2(base)))
    min_val_power = min(min_val_power, 0)
    bin_edges = base**np.array(range(min_val_power, max_val_power+1))
    # Step 2: we add in -∞ and +∞ to ensure all values fit into a bin.
    bin_edges = np.concatenate(([-np.inf], bin_edges, [np.inf]))
    # Step 3: bin the values in the column
    df["bin"] = pd.cut(df[col], bins=bin_edges, include_lowest=True)
    # Step 4: Compute bin_left, bin_right, bin_mid, and bin_power
    # Get lower and upper bounds of the bins
    df["bin_left"] = df["bin"].apply(lambda x: x if x is np.nan else x.left)
    df["bin_right"] = df["bin"].apply(lambda x: x if x is np.nan else x.right)
    # Get midpoints of the bin values
    df["bin_mid"] = df["bin"].apply(lambda x: np.mean([x.left, x.right]))
    # A convention we adopt is to use the left edge of the bin as the
    # reference bin value. Consistent with that convention, when we report the
    # power of each bin, we use the integer floor of the power of the left edge.
    df["bin_power"] = df["bin_left"].apply(lambda x: np.floor(np.log2(x) / np.log2(base))
                                           if np.isfinite(x) and x > 0 else np.nan)
    df["bin_power"] = df["bin_power"].apply(lambda x: x if np.isnan(x) else int(x))
    return df

dropped_items_warning=f"""Several items in `results` are not suitable for conversion to
a dataframe. This may be because they are not numpy arrays or because they
are not the same size as the other items. 

The following items were dropped: """
def results_to_dataframe_egt(results:dict, # A dictionary containing items from `ModelTypeEGT`.
                             suppress:bool=True, # Supress the dropped items warning
                             verbose:bool=False,
                            ):
    """Convert results to a dataframe, keeping only items which are valid for
    a dataframe to have."""
    flat_results = {k:v
                    for k,v in results.items()
                    if (isinstance(v, np.ndarray)
                        and not v.ndim > 1)}
    # Ideally, we can label the columns with the strategy names
    # However, this assumes that we are provided with either a strategy_set or
    # a recurrent_states value in the results dictionary which contains a
    # a number of labels equal to the number of columns in the ergodic array.
    assert 'ergodic' in results.keys()
    n_states = results['ergodic'].shape[1]
    if len(results.get('strategy_set', [])) == n_states:
        states = results['strategy_set']
    elif len(results.get('recurrent_states', [])) == n_states:
        states = results['recurrent_states']
    else:
        states = [f"state_{i}" for i in range(n_states)]
    for i, state in enumerate(states):
        if "ergodic" in list(results.keys()):
            flat_results[state + "_frequency"] = results['ergodic'][:,i]
    dropped_items = [k for k in results.keys() if k not in flat_results]
    if (len(dropped_items)>0 and not suppress):
        print(f"{dropped_items_warning} {dropped_items}")
    if verbose:
        print("Array shapes", 
              {k: results[k].shape
               for k,v in results.items() if isinstance(v, np.ndarray)})
        # Test that all entries in result_sums are close to 1
        result_sums = np.sum(results['ergodic'], axis=-1)
        assert np.allclose(result_sums, 1, atol=1e-10)
    return pd.DataFrame(flat_results)    

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
    # a list of the recurrent_states where that player employs that strategy
    strat_states_mapping = {}
    for player_index in range(n_players):
        player_strats = np.unique([state.split("-")[::-1][player_index]
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
    return a new list where only the first strategy for each player is kept.
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

def process_dsair_data(data):
    """Process DSAIR model results dataframe."""
    data['pr'] = np.round(1 - data['p'].values, 2)
    data['s'] = np.round(data['s'].values, 2)
    return data

def process_ai_trust_dataframe(data):
    """Process AI Trust model results dataframe."""
    # Ensure we can easily slice the dataframe by the Eps column
    data['Eps'] = np.round(data['Eps'].values, 2)
    return data
