import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def create_model(params):
    """
    Create a replicator dynamics model function from a parameter dictionary.
    
    Parameters:
        params: dict
            Dictionary containing the model parameters with keys:
            - 'bU'  : Parameter for F₁
            - 'eps' : Epsilon parameter for F₁
            - 'cP'  : Parameter for F₂
            - 'u'   : Parameter for F₂
            - 'cR'  : Parameter for F₃
            - 'v    '   : Parameter for F₃
    
    Returns:
        A function that computes the derivatives [dx/dt, dy/dt, dz/dt] given t and state [x, y, z].
    """
    def model(t, state):
        x, y_val, z = state  # Unpack the state variables
        # Compute derivatives according to the provided ODEs
        dxdt = x * (1 - x) * params["bU"] * (y_val + params["eps"] * (1 - y_val))
        dydt = y_val * (1 - y_val) * (-params["cP"] + params["u"] * x * z)
        dzdt = z * (1 - z) * (-params["cR"] + x * (1 - y_val) * (params["bfo"] - params["v"]))
        return [dxdt, dydt, dzdt]
    return model

def run_simulation(model, t_span, initial_conditions, t_eval=None):
    """
    Run the ODE simulation using SciPy's solve_ivp.
    
    Parameters:
        model: callable
            The ODE function (e.g., returned from create_model).
        t_span: tuple
            Tuple specifying the start and end times, e.g., (0, 500).
        initial_conditions: list or array
            Initial values for [x, y, z].
        t_eval: array-like, optional
            Time points at which to store the computed solution. If None, a linspace is used.
    
    Returns:
        solution: OdeResult object with attributes 't' (time points) and 'y' (solution values).
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)
    
    solution = solve_ivp(fun=model, t_span=t_span, y0=initial_conditions, t_eval=t_eval, method='RK45')
    return solution

def plot_simulation(solution, labels=['x', 'y', 'z']):
    """
    Plot the simulation results.
    
    Parameters:
        solution: OdeResult object
            The solution returned by run_simulation.
        labels: list of str
            Labels for each of the state variables.
    """
    plt.figure(figsize=(10, 6))
    for i in range(solution.y.shape[0]):
        plt.plot(solution.t, solution.y[i], label=labels[i])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Replicator Dynamics Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Default Example ---
if __name__ == '__main__':
    # Define default parameters
    params = {
        "bU": 4.0,
        "eps": 0.01,
        "cP": 0.5,
        "u": 1.5,
        "cR": 0.5,
        "bfo": 3.0,
        "v": 1.5
    }
    
    # Generate the model function using the parameter dictionary
    model = create_model(params)
    
    # Set the simulation parameters: time span and initial conditions for [x, y, z]
    t_span = (0, 100)
    initial_conditions = [0.5, 0.5, 0.5]  # example initial values
    
    # Run the simulation
    solution = run_simulation(model, t_span, initial_conditions)
    
    # Plot the results
    plot_simulation(solution)
