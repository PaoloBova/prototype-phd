"""
MCDEV Model Exploration with Pydantic, Single-Argument Functions, and Protocols
-------------------------------------------------------------------------------

This script demonstrates rewriting the MCDEV model exploration code with the
coding patterns recommended in coding_patterns.md:

1. Use Pydantic for data validation.
2. Single-argument functions (accept a single Pydantic model).
3. Explicit destructuring in each function.
4. Protocols for scenario definitions.
5. Minimal coupling and flexible design.

Additionally, it now includes an iterative corner-solution procedure to properly
handle translated CES (MCDEV) demands when multiple goods may drop to zero.

-------------------------------------------------------------
USAGE EXAMPLE:
    python mcdev_model.py

(Or incorporate these classes and functions in your own library or notebook.)
-------------------------------------------------------------
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Protocol, runtime_checkable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # For violin plots

# ---------------------------
# 1. Define Protocol for Price Scenarios
# ---------------------------
@runtime_checkable
class PriceScenarioFunc(Protocol):
    """Protocol for any callable that, given a k-vector array, returns a price array."""
    def __call__(self, k_vec: np.ndarray) -> np.ndarray:
        ...

# ---------------------------
# 2. Pydantic Models for Configuration
# ---------------------------
class ScenarioConfig(BaseModel):
    """Configuration for generating price vectors."""
    scenario_name: str = Field(..., description="Name of the scenario, e.g. 'Constant', 'Increasing', etc.")
    scenario_func: PriceScenarioFunc
    K: int = Field(30, description="Number of items.")

    class Config:
        arbitrary_types_allowed = True  # Permit storing the scenario_func callable

class DemandConfig(BaseModel):
    """Configuration model for computing optimal demands (closed-form)."""
    scenario_config: ScenarioConfig
    B: float = Field(..., description="Budget level.")
    p: np.ndarray = Field(..., description="Price array of length K.")
    psi: np.ndarray = Field(..., description="Psi array of length K.")
    gamma: np.ndarray = Field(..., description="Gamma array of length K.")
    alpha: float = Field(..., description="Alpha parameter in MCDEV model.")
    tol: float = Field(1e-8, description="Tolerance for dropping negative demands.")
    max_iter: int = Field(100, description="Maximum iterations before error.")

    class Config:
        arbitrary_types_allowed = True  # We allow np.ndarray

    @field_validator("p", "psi", "gamma")
    def check_array_lengths(cls, v, info):
        data = info.data
        if "p" in data and isinstance(v, np.ndarray) and len(v) != len(data["p"]):
            raise ValueError("All arrays p, psi, gamma must have the same length.")
        return v

class PlotConfig(BaseModel):
    """Configuration model for generating plots of the MCDEV allocations."""
    plot_group: dict = Field(..., description="Group that is being plotted, e.g. 'Alpha:0', 'Scenario:Increasing'.")
    df: pd.DataFrame = Field(..., description="DataFrame containing the computed allocations.")
    y_var: str = Field("Demand", description="Y-axis variable for plotting.")
    
    class Config:
        arbitrary_types_allowed = True

# ---------------------------
# 3. Utility Functions
# ---------------------------
def compute_theta(alpha: float) -> float:
    """Compute theta = 1 / (alpha - 1). Use theta = -1.0 when alpha=0 (Cobb-Douglas limit)."""
    if alpha == 0:
        return -1.0
    else:
        return 1.0 / (alpha - 1)


def compute_lambda(config: DemandConfig, theta: float) -> float:
    """Compute the Lagrange multiplier lambda for the MCDEV closed-form expression."""
    B = config.B
    p = config.p
    gamma = config.gamma
    psi = config.psi

    numerator = B + np.sum(p * gamma)
    denominator = np.sum(gamma * (psi ** (-theta)) * (p ** (1 + theta)))

    lambda_theta = numerator / denominator
    return lambda_theta ** (1.0 / theta)


def compute_lambda_iterative(cfg: DemandConfig, theta: float, indices: np.ndarray) -> float:
    """Compute lambda for the subset of goods in `indices`, using the corner-solution formula."""
    B = cfg.B
    p = cfg.p
    gamma = cfg.gamma
    psi = cfg.psi

    numerator = B + np.sum(p[indices] * gamma[indices])
    denominator = np.sum(
        gamma[indices] * (psi[indices] ** (-theta)) * (p[indices] ** (1 + theta))
    )
    lam_theta = numerator / denominator
    return lam_theta ** (1.0 / theta)


def compute_optimal_demands(cfg: DemandConfig) -> np.ndarray:
    """Iterative approach for MCDEV demands with corner solutions.

    1. Start with all items.
    2. Compute lambda on that set.
    3. Drop items with negative demands.
    4. Repeat until stable or max_iter is reached.
    """
    p = cfg.p
    psi = cfg.psi
    gamma = cfg.gamma
    alpha = cfg.alpha
    tol = cfg.tol

    K = len(p)
    theta = compute_theta(alpha)

    # Start with full set of indices
    S = np.arange(K)
    iteration = 0

    while True:
        iteration += 1
        if iteration > cfg.max_iter:
            raise RuntimeError("Max iterations exceeded in iterative corner-solution procedure.")
        lam = compute_lambda_iterative(cfg, theta, S)
        # Candidate demands
        x_candidates = gamma[S] * ((lam * p[S] / psi[S]) ** theta - 1.0)
        # Keep items with x_k >= -tol
        S_new = S[x_candidates >= -tol]
        if np.array_equal(S_new, S):
            # Converged
            break
        S = S_new

    # Final x
    x = np.zeros(K, dtype=float)
    x_star_pos = gamma[S] * ((lam * p[S] / psi[S]) ** theta - 1.0)
    x[S] = np.maximum(x_star_pos, 0.0)
    return x


# ---------------------------
# 4. Scenario Functions
# ---------------------------
def scenario_constant(k_vec: np.ndarray, a: float=1.0, b: float=0.2) -> np.ndarray:
    """p_k = a + b * k (linear increase)."""
    return a + b * k_vec


def scenario_increasing(k_vec: np.ndarray, a: float=1.0, b: float=0.05) -> np.ndarray:
    """p_k = a + b * (k^2) (quadratic)."""
    return a + b * (k_vec ** 2)


def scenario_diminishing(k_vec: np.ndarray, a: float=1.0, b: float=0.5) -> np.ndarray:
    """p_k = a + b * log(1 + k)."""
    return a + b * np.log(1.0 + k_vec)

def scenario_exponential(k_vec: np.ndarray, a: float=1.0, b: float=1.0) -> np.ndarray:
    """p_k = a * exp(b * k)."""
    return a * np.exp2(b * k_vec)

# ---------------------------
# 5. Run the Model
# ---------------------------

def compute_allocations(scenarios: List[DemandConfig]) -> pd.DataFrame:
    """
    Compute allocations for a given scenario and alpha over specified budget values,
    and return a long-format dataframe.
    """
    allocations = [compute_optimal_demands(cfg) for cfg in scenarios]
    allocations = np.array(allocations)
    long_data = []
    for i, scenario in enumerate(scenarios):
        for k in range(scenario.scenario_config.K):
            long_data.append({'Item': k + 1,
                              'Demand': allocations[i, k],
                              'Expenditure': allocations[i, k] * scenario.p[k],
                              'Price': scenario.p[k],
                              'Psi': scenario.psi[k],
                              'Gamma': scenario.gamma[k],
                              'Alpha': scenario.alpha,
                              "Scenario": scenario.scenario_config.scenario_name,
                              'Budget': scenario.B})
    df = pd.DataFrame(long_data)
    return df

def plot_allocations(config: PlotConfig) -> object:
    """
    Plot heatmap and violin plots of optimal allocations using precomputed dataframe in config.df.
    """
    df = config.df
    # Create a faceted bar chart with a facet row for each Budget,
    # retaining hues for each Budget.
    # Make sure wrap facets if there are too many budgets.
    # Use seaborn's catplot for faceting
    y_var=config.y_var
    g = sns.catplot(
         data=df,
         x="Item",
         y=y_var,
         hue="Budget",
         col="Budget",
         kind="bar",
         height=3,
         aspect=1.5,
         palette="viridis",
         col_wrap=4,
         sharey=False,
    )
    g.set_axis_labels("Item", y_var)
    g.set_titles("Budget = {col_name}")
    group_string = [f"{group_var}={group_val}".replace(" ", "_")
                    for group_var, group_val in config.plot_group.items()]
    group_string = " ".join(group_string)
    plot_title = f"Bar Chart of {y_var} by Item and Budget: {group_string}"
    g.figure.suptitle(plot_title, y=1.02)
    # Remove the legend from the bar plots
    g._legend.remove()
    plt.tight_layout()
    return g.figure  # Return the figure object