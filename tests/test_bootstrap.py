"""
Test module for bootstrap.py, revised to demonstrate Pydantic usage and property-based testing.

We illustrate:
1. Generating random BootstrapConfig objects with Hypothesis.
2. Generating random small DataFrames.
3. Ensuring the system runs end-to-end without errors.

Because Hypothesis does not provide a built-in strategy for creating DataFrames, we define a simple approach:
- Generate numeric columns as lists of floats.
- Assemble them into a DataFrame.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Optional

from hypothesis import given, strategies as st, settings
from pydantic import BaseModel, Field

import prototype_phd.methods.bootstrap as bootstrap


# We define a simple analysis function that returns the mean of each column
def analysis_mean(indices: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    arr = data.to_numpy()
    sub = arr[indices, :]  # use 2D indexing to select rows for 3D array
    means = np.mean(sub, axis=1)
    col_names = [f"{c}_mean" for c in data.columns]
    results = dict(zip(col_names, means.T))
    return pd.DataFrame(results)


# Another analysis function for standard deviation
def analysis_std(indices: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    arr = data.to_numpy()
    sub = arr[indices, :]  # use 2D indexing to select rows for 3D array
    stdevs = np.std(sub, axis=1, ddof=1)
    col_names = [f"{c}_std" for c in data.columns]
    results = dict(zip(col_names, stdevs.T))
    return pd.DataFrame(results)

# A more complex analysis function that combines x and y columns
def analysis_combine_x_y(indices: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    arr = data[["col_1", "col_2"]].to_numpy()
    sub = arr[indices, :]  # use 2D indexing to select rows for 3D array
    new_data = sub[:, :, [1]] + sub[:, :, [0]]
    new_data = np.mean(new_data, axis=1)
    results = dict(zip(["col_1+col_2"], new_data.T))
    return results

# A function to compute logistic regression coefficients for y ~ x
def analysis_logistic_regression(indices: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    from prototype_phd.stats import robust_logistic_fit
    arr = data[["col_1", "col_2"]].to_numpy()
    sub = arr[indices, :]  # use 2D indexing to select rows for 3D array
    X = sub[..., :-1]  # all columns except the last one
    y = sub[..., -1]   # the last column
    # Fit a logistic regression model for each bootstrap sample
    n_samples = X.shape[0]
    coefficients = np.zeros((n_samples, X.shape[-1]+1))
    for i in range(n_samples):
        results = robust_logistic_fit(X[i, ...], y[i, ...])
        # Extract the coefficients
        coefficients[i, :] = results.params
    # Create a DataFrame with the coefficients
    results = dict(zip([f"coeff_{i}" for i in range(len(coefficients.T))], coefficients.T))
    return pd.DataFrame(results)

# Note: Some functions like the above don't really benefit from the multi-indexing
# and are more straightforward with 2D arrays. However, we keep the
# multi-indexing for consistency with the rest of the code.
# In principle, we could write our own logistic regression implementation to
# allow for numpy broadcasting for multiple regressions.

class BootstrapDataExample(BaseModel):
    df: pd.DataFrame = Field(..., description="Input DataFrame for bootstrap analysis.")
    bootstrap_config: bootstrap.BootstrapConfig = Field(
        ..., description="Configuration for bootstrap analysis."
    )
    
    class Config:
        
        arbitrary_types_allowed = True
        # Allow DataFrame to be used in Pydantic model
        json_encoders = {
            pd.DataFrame: lambda v: v.to_dict(orient='records')
        }


# We'll define a strategy for generating random config.
# We'll keep n_bootstrap fairly small to avoid huge tests.
@st.composite
def bootstrap_config_strat(draw):
    n_bootstrap = draw(st.integers(min_value=1, max_value=20))
    sample_size = draw(st.one_of(
        st.none(),
        st.integers(min_value=1, max_value=20)
    ))
    random_state = draw(st.one_of(
        st.none(),
        st.integers(min_value=0, max_value=9999)
    ))
    weighted = draw(st.booleans())
    
    analysis_funcs = draw(st.lists(
        st.sampled_from([
            analysis_mean,
            analysis_std
        ]), min_size=1, max_size=3))
    # Ensure the analysis functions are unique
    analysis_funcs = list(set(analysis_funcs))

    return bootstrap.BootstrapConfig(
        n_bootstrap=n_bootstrap,
        sample_size=sample_size,
        random_state=random_state,
        weighted=weighted,
        analysis_funcs=analysis_funcs
    )


# We'll define a strategy for generating small numeric DataFrames
@st.composite
def small_numeric_dataframe(draw):
    n_rows = draw(st.integers(min_value=1, max_value=20))
    n_cols = draw(st.integers(min_value=1, max_value=5))

    data = {}
    for c in range(n_cols):
        colname = f"col_{c}"
        col_values = draw(
            st.lists(
                st.floats(min_value=-100, max_value=100, allow_infinity=False, allow_nan=False),
                min_size=n_rows,
                max_size=n_rows
            )
        )
        data[colname] = col_values
    df = pd.DataFrame(data)
    return df


# We'll define a strategy for generating small numeric DataFrames with missing values
@st.composite
def small_numeric_dataframe_with_missing(draw):
    n_rows = draw(st.integers(min_value=1, max_value=20))
    n_cols = draw(st.integers(min_value=1, max_value=5))
    data = {}
    for c in range(n_cols):
        colname = f"col_{c}"
        # Strategy that allows NaN values: don't restrict the range.
        col_values = draw(
            st.lists(
                st.one_of(
                    st.floats(allow_nan=True, allow_infinity=False),
                    st.none()
                ),
                min_size=n_rows,
                max_size=n_rows
            )
        )
        data[colname] = col_values
    return pd.DataFrame(data)


@st.composite
def logistic_data_strategy(draw):
    # Generate true logistic parameters
    beta0 = draw(st.floats(min_value=-1, max_value=1))
    beta1 = draw(st.floats(min_value=-2, max_value=2))
    n = draw(st.integers(min_value=150, max_value=250))
    # Generate predictor values
    x_list = draw(st.lists(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    x = np.array(x_list)
    linear_term = beta0 + beta1 * x
    probs = 1 / (1 + np.exp(-linear_term))
    # Use a fixed seed for consistency
    rng = np.random.default_rng(42)
    y = rng.binomial(1, probs)
    df = pd.DataFrame({"col_1": x, "col_2": y})
    return df, beta0, beta1


@given(
    df=small_numeric_dataframe(),
    bootstrap_config=bootstrap_config_strat()
)
def test_run_bootstrap_property(df, bootstrap_config):
    input_data = BootstrapDataExample(df=df, bootstrap_config=bootstrap_config)
    results = bootstrap.run_bootstrap(input_data)

    # Basic checks: results should not be empty
    # as soon as n_bootstrap >=1 and n_rows>0
    if len(df) > 0 and bootstrap_config.n_bootstrap >= 1:
        assert len(results) > 0


@settings(deadline=None)
@given(logistic_data=logistic_data_strategy())
def test_bootstrap_logistic_regression_estimates(logistic_data):
    df, true_beta0, true_beta1 = logistic_data
    cfg = bootstrap.BootstrapConfig(
        n_bootstrap=20,
        sample_size=len(df),
        random_state=123,
        weights=None,
        analysis_funcs=[analysis_logistic_regression]
    )
    # Use the concrete BootstrapDataExample to wrap the inputs.
    input_data = BootstrapDataExample(df=df, bootstrap_config=cfg)
    results = bootstrap.run_bootstrap(input_data)
    
    # The analysis function prefixes the returned columns with "analysis_logistic_regression_"
    col0 = results["coeff_0"].mean()
    col1 = results["coeff_1"].mean()
    
    tol = 1 # tolerance for the estimates
    assert abs(col0 - true_beta0) < tol, f"Intercept estimate {col0} not within {tol} of true {true_beta0}"
    assert abs(col1 - true_beta1) < tol, f"Coefficient estimate {col1} not within {tol} of true {true_beta1}"


# We can keep some smaller direct tests for coverage

def test_generate_bootstrap_indices_basic():
    dataset_size = 10
    n_bootstrap = 5
    # Minimal usage without weighting
    cfg = bootstrap.BootstrapConfig(
        n_bootstrap=n_bootstrap,
        analysis_funcs=[analysis_mean]  # not used here
    )
    input_data = BootstrapDataExample(
        df=pd.DataFrame({'x': range(dataset_size)}),
        bootstrap_config=cfg
    )
    indices_list = bootstrap.generate_bootstrap_indices(input_data)
    assert len(indices_list) == n_bootstrap
    for indices in indices_list:
        assert len(indices) == dataset_size
        assert np.all(indices < dataset_size)


def test_bootstrap_summary():
    data = {
        'analysis_func': ["a", "a", "b", "b"],
        'value': [10, 12, 5, 7]
    }
    df = pd.DataFrame(data)
    summary = bootstrap.bootstrap_summary(df, value_col='value', group_cols=['analysis_func'])
    assert 'mean' in summary.columns
    assert 'std' in summary.columns
    assert 'stderr' in summary.columns
    row_a = summary[summary['analysis_func'] == "a"].iloc[0]
    assert row_a['mean'] == 11


def test_bootstrap_confidence_interval():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5, scale=1, size=1000)
    lower, upper = bootstrap.bootstrap_confidence_interval(data)
    # With 1000 normal samples, the 95% CI should contain ~5
    assert lower < 5 < upper


def test_bootstrap_normality_test():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=1, size=500)
    # Usually passes, but might fail rarely
    result = bootstrap.bootstrap_normality_test(data)
    # Just check it runs


def test_bootstrap_ttest():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5, scale=1, size=500)
    result = bootstrap.bootstrap_ttest(data, hypothesized_mean=5)
    # Typically won't reject. We won't assert either way, just no error.


def test_bootstrap_logistic_regression():

    # Create synthetic data with required columns for logistic regression.
    df = pd.DataFrame({
        "col_1": np.random.normal(0, 1, 100),
        "col_2": np.random.binomial(1, 0.5, 100)
    })
    
    # Set up a BootstrapConfig using only the logistic regression analysis function.
    cfg = bootstrap.BootstrapConfig(
        n_bootstrap=10,
        sample_size=100,
        random_state=42,
        weights=None,
        analysis_funcs=[analysis_logistic_regression]
    )
    input_data = BootstrapDataExample(df=df, bootstrap_config=cfg)
    results = bootstrap.run_bootstrap(input_data)
    
    # Verify that the output DataFrame contains the logistic regression coefficient columns.
    expected_cols = [f"coeff_{i}" for i in range(2)]
    for col in expected_cols:
        assert col in results.columns
    # Verify that the output DataFrame has the expected number of rows.
    assert len(results) == cfg.n_bootstrap
    # Check that the coefficients are numeric.
    for col in expected_cols:
        assert pd.api.types.is_numeric_dtype(results[col])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
