import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import plotly.graph_objects as go
import random
import tqdm
import uuid

dropped_items_warning=f"""Several items in `results` are not suitable for conversion to
a dataframe. This may be because they are not numpy arrays or because they
are not the same size as the other items. 

The following items were dropped: """
def results_to_dataframe_egt(results:dict, # A dictionary containing items from `ModelTypeEGT`.
                             suppress:bool=True, # Supress the dropped items warning
                            ):
    """Convert results to a dataframe, keeping only items which are valid for
    a dataframe to have."""
    flat_results = {k:v
                    for k,v in results.items()
                    if (isinstance(v, np.ndarray)
                        and not v.ndim > 1)}
    for i, strategy in enumerate(results.get('recurrent_states', 
                                             results.get('strategy_set', []))):
        if "ergodic" in list(results.keys()):
            flat_results[strategy + "_frequency"] = results['ergodic'][:,i]
    dropped_items = [k for k in results.keys() if k not in flat_results]
    if (len(dropped_items)>0 and not suppress):
        print(f"{dropped_items_warning} {dropped_items}")
    return pandas.DataFrame(flat_results)    

def process_dsair_data(data):
    """Process DSAIR model results dataframe."""
    data['pr'] = np.round(1 - data['p'].values, 2)
    data['s'] = np.round(data['s'].values, 2)
    return data

def is_plain_word(word):
    return word.isalpha()

def generate_random_phrase(words, num_words=3):
    return '_'.join(random.sample(words, num_words))

def create_id(path_to_data='/usr/share/dict/words', verbose=True):
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
    
def save_data(data, folder_name=None):
    # Usage:
    # data = {'df': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 'dict': {'key1': 'value1', 'key2': 'value2'}}
    # save_data(data, 'my_folder')
    # If no folder name is provided, generate a random one
    if folder_name is None:
        folder_name = f'data/{create_id()}'
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Iterate over the items in the dictionary
    for key, value in data.items():
        # If the value is a DataFrame, save it as a CSV file
        if isinstance(value, pandas.DataFrame):
            value.to_csv(os.path.join(folder_name, f'{key}.csv'), index=False)
        # If the value is a dictionary, save it as a JSON file
        elif isinstance(value, dict):
            with open(os.path.join(folder_name, f'{key}.json'), 'w') as f:
                json.dump(value, f)

def run_all_simulations(param_list: list,
                        simulation_fn:callable=None,
                        plotting_fn:callable=None,
                        simulation_dir:str="data",
                        plot_dir:str="plots",):
    """
    Iterate over each parameter dictionary, run the simulation, and save the results.

    Parameters:
    - param_list: A list of dictionaries, each containing a set of parameter values.
    """
    
    # Check if the output directory exists. If not, create it.
    if simulation_dir and not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    figs = []
    simulation_results = []
    # Construct a unique directory name
    simulation_id = create_id()
    os.makedirs("/".join([simulation_dir, simulation_id]))
    os.makedirs("/".join([plot_dir, simulation_id]))
    for idx, parameters in tqdm.tqdm(enumerate(param_list)):
        if simulation_fn is not None:
            df = simulation_fn(parameters)
            df["simulation_id"] = simulation_id
            df["model_id"] = idx
            # Save the dataframe to CSV
            filename = f"dataframe_{simulation_id}_{idx}.csv"
            filepath = os.path.join(simulation_dir, simulation_id, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved file: {filepath}")
            simulation_results.append(df)
        if plotting_fn is not None:
            fig = plotting_fn(parameters)
            if len(fig) > 1:
                for i, f in enumerate(fig):
                    # Save the figure
                    filename = f"plot_{simulation_id}_fig_{i}_{idx}.png"
                    filepath = os.path.join(plot_dir, simulation_id, filename)
                    if f is not None:
                        if isinstance(f, go.Figure):
                            # This is a Plotly figure
                            f.write_image(filepath)
                        elif isinstance(f, plt.Figure):
                            # This is a Matplotlib figure
                            f.savefig(filepath)
                            plt.close(f)  # Close the figure to free up memory
                        print(f"Saved file: {filepath}")
                    figs.append(f)
                    
            else:
                # Save the figure
                filename = f"plot_{simulation_id}_{idx}.png"
                filepath = os.path.join(plot_dir, simulation_id, filename)
                if fig is not None:
                    if isinstance(fig, go.Figure):
                        # This is a Plotly figure
                        fig.write_image(filepath)
                    elif isinstance(fig, plt.Figure):
                        # This is a Matplotlib figure
                        fig.savefig(filepath)
                        plt.close(fig)  # Close the figure to free up memory
                    print(f"Saved file: {filepath}")
                figs.append(fig)
        
    return figs, simulation_results
