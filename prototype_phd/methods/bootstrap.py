"""
A Python module to decouple bootstrapping utilities from the rest of the project.


This module provides:

1. A memory-efficient function to generate bootstrap samples (via indices),
   optionally with weights.
2. A framework to apply multiple user-defined analysis functions to each
   bootstrap sample.
3. Utility functions for analyzing results from the bootstrap, such as
   computing summary statistics, confidence intervals, normality tests, etc.

Example Usage:

    import pandas as pd

    def compute_mean(indices, df):
        # Return a DataFrame containing the mean of df for each column.
        sub = df.iloc[indices]
        return pd.DataFrame({"mean_": sub.mean()}, index=[0])

    def compute_std(indices, df):
        sub = df.iloc[indices]
        return pd.DataFrame({"std_": sub.std()}, index=[0])

    data = pd.DataFrame({
        "col1": [1,2,3,4,5],
        "col2": [2,3,4,5,6]
    })

    config = {
        "n_bootstrap": 1000,
        "sample_size": None,  # same size as original dataset
        "weights": None,      # optional weighting array or Series
        "random_state": 42,
        "analysis_funcs": [compute_mean, compute_std]
    }
    config = BootstrapConfig(**config)
    input_data = BootstrapInput(df=data, config=config)
    results = run_bootstrap(input_data)
    print(results.head())
    # Typically, you'd do further analysis (e.g. confidence intervals) on these results.
    # For example, to compute a summary:
    summary = bootstrap_summary(results, value_col="mean_")

"""

import numpy as np
import pandas as pd
import prototype_phd.stats as stats
from pydantic import BaseModel, Field, field_validator
from scipy.stats import sem, ttest_1samp, normaltest
from typing import Callable, List, Optional, Protocol, runtime_checkable

class BootstrapConfig(BaseModel):
    n_bootstrap: int = Field(..., description="Number of bootstrap replications.")
    analysis_funcs: List[Callable[[np.ndarray, pd.DataFrame], pd.DataFrame]] = Field(
        ..., description="List of callables each taking (indices, df) -> pd.DataFrame"
    )
    sample_size: Optional[int] = Field(
        None,
        description="Size of each bootstrap sample. If None, defaults to dataset size."
    )
    weights: Optional[np.ndarray] = Field(
        None,
        description="Optional weighting array for sampling, must sum to 1 if provided."
    )
    random_state: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )
    hierarchy_columns: Optional[List[str]] = Field(
        None,
        description="List of column names that define hierarchical groups. If provided, bootstrapping will be performed first over groups, then within groups."
    )

    class Config:
        arbitrary_types_allowed = True
    
    
    @field_validator('weights')
    @classmethod
    def check_weights(cls, w):
        if w is not None:
            s = float(w.sum())
            if not np.isclose(s, 1.0):
                raise ValueError(f"weights must sum to 1. Received sum={s}.")
            if np.any(w < 0):
                raise ValueError("weights cannot be negative.")
        return w

    @field_validator('analysis_funcs')
    @classmethod
    def check_nonempty_funcs(cls, funcs):
        if len(funcs) == 0:
            raise ValueError("analysis_funcs must contain at least one callable.")
        return funcs
    
    
    @field_validator('analysis_funcs')
    @classmethod
    def check_all_unique(cls, funcs):
        if len(funcs) != len(set(funcs)):
            raise ValueError("analysis_funcs must contain unique callables.")
        return funcs


@runtime_checkable
class BootstrapData(Protocol):
    df: pd.DataFrame
    bootstrap_config: BootstrapConfig
    
class BootstrapDataInput(BaseModel):
    df: pd.DataFrame = Field(..., description="DataFrame containing the data to be bootstrapped.")
    bootstrap_config: BootstrapConfig = Field(
        ..., description="Configuration for the bootstrap process."
    )

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types in the model
        # validate_assignment = True  # Enable validation on assignment
        # use_enum_values = True  # Use enum values for validation


def generate_bootstrap_indices(input_data: BootstrapData) -> List[np.ndarray]:
    """
    Generate a list of bootstrap indices based on the configuration.

    This function interprets the dataset size from input_data.df,
    and uses the fields from input_data.config.
    """
    df = input_data.df
    config = input_data.bootstrap_config

    dataset_size = len(df)
    sample_size = config.sample_size if config.sample_size is not None else dataset_size

    rng = np.random.default_rng(config.random_state)

    indices_list = []
    for _ in range(config.n_bootstrap):
        chosen = rng.choice(
            dataset_size,
            size=sample_size,
            replace=True,
            p=config.weights
        )
        indices_list.append(chosen)

    return indices_list

def generate_hierarchical_bootstrap_indices(input_data: BootstrapData) -> List[dict]:
    """
    Generate a list of bootstrap indices for hierarchical sampling.
    
    For each bootstrap replicate, sample groups with replacement 
    (using the first hierarchy column) and then sample indices within each group.
    
    Returns a list where each replicate is a dictionary mapping group names to sampled indices.
    """
    df = input_data.df
    config = input_data.bootstrap_config
    hierarchy = config.hierarchy_columns
    if not hierarchy or len(hierarchy) == 0:
        # Fallback to standard bootstrap indices
        indices_list = generate_bootstrap_indices(input_data)
        return [{"all": ind} for ind in indices_list]

    group_col = hierarchy[0]
    groups = df[group_col].unique()
    rng = np.random.default_rng(config.random_state)
    nested_indices_list = []
    
    for _ in range(config.n_bootstrap):
        # Sample groups with replacement
        chosen_groups = rng.choice(groups, size=len(groups), replace=True)
        indices_for_replicate = {}
        for group in chosen_groups:
            group_data = df[df[group_col] == group]
            group_indices = group_data.index.to_numpy()
            sample_size = config.sample_size if config.sample_size is not None else len(group_indices)
            # Sample within the group with replacement
            sampled_indices = rng.choice(group_indices, size=sample_size, replace=True)
            # Append or initialize the sampled indices for the group
            if group in indices_for_replicate:
                indices_for_replicate[group] = np.concatenate([indices_for_replicate[group], sampled_indices])
            else:
                indices_for_replicate[group] = sampled_indices
        nested_indices_list.append(indices_for_replicate)
    return nested_indices_list


def apply_bootstrap_functions(
    indices_list: List[np.ndarray],
    input_data: BootstrapData
) -> pd.DataFrame:
    """
    Apply a list of analysis functions to each set of bootstrap indices.

    Each function receives (indices, df) and must return a pd.DataFrame.
    We'll concatenate all returned DataFrames for all functions and replicates.
    """
    df = input_data.df
    analysis_funcs = input_data.bootstrap_config.analysis_funcs
    df_results = pd.DataFrame(index=np.arange(len(indices_list)))
    df_results["replicate"] = np.arange(len(indices_list))
    assert len(indices_list) > 0, f"No bootstrap indices generated. Please check your configuration. Indices list: {indices_list}"
    indices_array = np.array(indices_list)
    results = []
    for func in analysis_funcs:
        func_name = getattr(func, "__name__", "<anonymous>")
        result = func(indices_array, df)
        assert isinstance(result, pd.DataFrame), f"Applying bootstrap functions failed. Function {func_name} must return a DataFrame."
        assert len(result) == len(indices_list), f"Applying bootstrap functions failed. Function {func_name} must return a DataFrame with one row per bootstrap replicant. Returned {result} instead."
        results.append(result)
    # Concatenate all results into a single DataFrame
    all_result_columns = [col for result in results for col in result.columns]
    assert len(all_result_columns) == len(set(all_result_columns)), "Duplicate column names found in bootstrap results."
    df_results = pd.concat(results, axis=1)

    return df_results


def run_bootstrap(input_data: BootstrapData) -> pd.DataFrame:
    """
    Orchestrate the entire bootstrap pipeline:
    - Generate bootstrap indices
    - Apply analysis funcs

    Returns a DataFrame with all computed results.
    """
    indices_list = generate_bootstrap_indices(input_data)
    results = apply_bootstrap_functions(indices_list, input_data)
    return results


# -----------------------------------------------------------------------------
# Below: Example analysis functions for bootstrapping.
# -----------------------------------------------------------------------------

def analysis_logistic_regression(indices: np.ndarray,
                                 data: pd.DataFrame,
                                 config: stats.LogRegConfig = stats.LogRegConfig(),
                                 x_cols: List[str] = ["x"],
                                 y_col: str = "y") -> pd.DataFrame:
    # X is an array with shape (n_samples, n_observations, n_features)
    X = data[x_cols].to_numpy()[indices, :]
    # y is an array with shape (n_samples, n_observations, 1)
    y = data[[y_col]].to_numpy()[indices, :]
    n_samples = X.shape[0]
    coefficients = np.zeros((n_samples, X.shape[-1] + 1))
    result_objs = []
    thresholds = []
    for i in range(n_samples):
        result_obj = stats.fit_logistic(X[i, ...], y[i, :, 0], config)
        result_objs.append(result_obj)
        coefficients[i, :] = result_obj.coeffs
        thresholds.append(stats.compute_threshold_from_result(result_obj))
    results_dict = dict(zip([f"coeff_{i}" for i in range(coefficients.shape[1])],
                            coefficients.T))
    results_dict["threshold"] = np.array(thresholds)
    results_dict["convergence"] = np.array([result.convergence for result in result_objs])
    results_dict["warning"] = np.array([result.warning for result in result_objs])
    return pd.DataFrame(results_dict)

# Note: Some functions like the above don't really benefit from the multi-indexing
# and are more straightforward with 2D arrays. However, we keep the
# multi-indexing for consistency with the rest of the code.
# In principle, we could write our own logistic regression implementation to
# allow for numpy broadcasting for multiple regressions.

def analysis_y_reliability(indices, df, x_col, y_col):
    """A bootstrap helper that for each value of x_col computes the
    y_pct reliability of the model where y_col is the binarized score."""
      # X is an array with shape (n_samples, n_observations)
    X = df[x_col].to_numpy()[indices]
    # y is an array with shape (n_samples, n_observations)
    Y = df[y_col].to_numpy()[indices]
    n_samples = indices.shape[0]
    all_x_set = np.unique(df[x_col].values)
    n_x_set = len(all_x_set)
    x_mapping = {x: i for i, x in enumerate(all_x_set)}
    estimates = np.zeros((n_samples, n_x_set))
    for i in range(n_samples):
        x_values, y_values = X[i, :], Y[i, :]
        
        # Compute the y_pct reliability for each value of x_col
        # This will be a new column in the DataFrame
        x_set = np.unique(x_values)
        for x in x_set:
            # Get the data for the current x value
            success_rate = np.mean(y_values[x_values == x])
            estimates[i, x_mapping[x]] = success_rate
    results_dict = dict(zip([f"estimate_{x}" for x in all_x_set], estimates.T))
    return pd.DataFrame(results_dict)

def analysis_weighted_sum(indices, df, x_col, y_col, level_weight_fn=None, info_weight_fn=None):
    """A bootstrap helper that computes a weighted sum
    of the success rates across x_col as measured by y_col."""
      # X is an array with shape (n_samples, n_observations)
    X = df[x_col].to_numpy()[indices]
    # y is an array with shape (n_samples, n_observations)
    Y = df[y_col].to_numpy()[indices]
    n_samples = indices.shape[0]
    all_x_set = np.unique(df[x_col].values)
    n_x_set = len(all_x_set)
    x_mapping = {x: i for i, x in enumerate(all_x_set)}
    estimates = np.zeros((n_samples, n_x_set))
    counts = np.zeros((n_samples, n_x_set))
    for i in range(n_samples):
        x_values, y_values = X[i, :], Y[i, :]
        
        # Compute the y_pct reliability for each value of x_col
        # This will be a new column in the DataFrame
        x_set = np.unique(x_values)
        for x in x_set:
            # Get the data for the current x value
            success_rate = np.mean(y_values[x_values == x])
            count = np.sum(x_values == x)
            counts[i, x_mapping[x]] = count
            estimates[i, x_mapping[x]] = success_rate
    results_dict = dict(zip([f"estimate_{x}" for x in all_x_set], estimates.T))
    # Compute the weighted sum of success rates across all x values
    if level_weight_fn is None:
        level_weight_fn = lambda x: 1.0
    level_weights = np.array([level_weight_fn(x) for x in all_x_set])
    if info_weight_fn is None:
        info_weights = np.ones((n_samples, n_x_set))
    else:
        info_weights = np.array([info_weight_fn(all_x_set[i], counts[:, i])
                                 for i in range(n_x_set)]).T
    assert info_weights.shape == (n_samples, n_x_set), f"info_weights shape mismatch: {info_weights.shape} != {(n_samples, n_x_set)}"
    weighted_sum = (info_weights * estimates) @ level_weights
    assert weighted_sum.shape == (n_samples,), f"weighted_sum shape mismatch: {weighted_sum.shape} != {(n_samples,)}"
    results_dict["weighted_sum"] = weighted_sum
    return pd.DataFrame(results_dict)


# -----------------------------------------------------------------------------
# Below: Additional utility functions for analyzing bootstrap results.
# -----------------------------------------------------------------------------

def bootstrap_summary(
    results: pd.DataFrame,
    value_col: str,
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute a simple summary (mean, std, stderr) for a given value column,
    optionally grouped by replicate or function or any other columns.
    """
    if group_cols is None:
        group_cols = []

    if len(results) == 0:
        return pd.DataFrame()

    grouped = results.groupby(group_cols)[value_col]
    summary_df = grouped.agg([
        ("mean", "mean"),
        ("std", "std"),
        ("stderr", sem)
    ]).reset_index()

    return summary_df


def bootstrap_confidence_interval(
    data: np.ndarray,
    alpha: float = 0.05
) -> tuple[float, float]:
    """
    Compute a basic percentile-based confidence interval for the data.
    """
    if data.size == 0:
        return (float('nan'), float('nan'))
    lower = np.percentile(data, 100 * (alpha / 2))
    upper = np.percentile(data, 100 * (1 - alpha / 2))
    return lower, upper


def bootstrap_normality_test(
    data: np.ndarray,
    alpha: float = 0.05
) -> bool:
    """
    Perform a normality test on the bootstrap estimates.

    Returns True if we fail to reject normality.
    """
    if data.size < 8:
        # normaltest requires at least 8 data points
        return True  # not enough data to reject
    stat, pvalue = normaltest(data)
    return pvalue > alpha


def bootstrap_ttest(
    data: np.ndarray,
    hypothesized_mean: float = 0.0,
    alpha: float = 0.05
) -> bool:
    """
    Perform a one-sample t-test on the bootstrap estimates to check if the mean differs.

    Returns True if fail to reject (mean == hypothesized_mean), else False.
    """
    if data.size < 2:
        return True  # Not enough data to reject
    stat, pvalue = ttest_1samp(data, popmean=hypothesized_mean)
    return pvalue > alpha
