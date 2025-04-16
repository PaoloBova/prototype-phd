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
from typing import Optional, List

from hypothesis import given, strategies as st, settings
from pydantic import BaseModel, Field

import prototype_phd.methods.bootstrap as bootstrap
from prototype_phd.stats import fit_logistic, LogRegConfig

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
    # Generate true logistic parameters using decimals for better precision
    beta0 = float(draw(
        st.decimals(min_value="-1", max_value="1", allow_nan=False, allow_infinity=False, places=2)
    ))
    beta1 = float(draw(
        st.decimals(min_value="-2", max_value="2", allow_nan=False, allow_infinity=False, places=2)
    ))
    n = draw(st.integers(min_value=150, max_value=200))
    # Generate predictor values as decimals then convert to floats
    x = np.linspace(-3, 3, n)
    linear_term = beta0 + beta1 * x
    probs = 1 / (1 + np.exp(-linear_term))
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
    stats_fn = lambda indices, data: bootstrap.analysis_logistic_regression(
        indices=indices,
        data=data,
        x_cols=["col_1"],
        y_col="col_2",
        config=LogRegConfig(engine="statsmodels", regularize=False)
    )
    cfg = bootstrap.BootstrapConfig(
        n_bootstrap=10,
        sample_size=100,
        random_state=42,
        weights=None,
        analysis_funcs=[stats_fn]
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

@pytest.mark.skip(reason="statsmodels regularization is not working as expected")
def test_bootstrap_regularized_logistic_regression():
    # For regularized logistic regression one does not expect consistency to the true parameter,
    # but rather that the bootstrap distribution centers on the full-sample regularized estimate.
    # Compute the full-sample regularized estimate:
    true_beta0 = 0.5
    true_beta1 = -1.25
    n = 10000
    np.random.seed(42)
    x = np.linspace(-3, 3, n)
    linear_term = true_beta0 + true_beta1 * x
    probs = 1 / (1 + np.exp(-linear_term))
    rng = np.random.default_rng(42)
    y = rng.binomial(1, probs)
    df = pd.DataFrame({"col_1": x, "col_2": y})
    
    # Compute the full-sample regularized estimate using fit_logistic.
    config_full = LogRegConfig(engine="statsmodels", regularize=True, alpha=1.0, L1_wt=1.0)
    X_full = df["col_1"].values.reshape(-1, 1)
    y_full = df["col_2"].values
    full_res = fit_logistic(X_full, y_full, config_full)
    full_coef0, full_coef1 = full_res.coeffs

    # Set up BootstrapConfig using the current regularized analysis function.
    cfg = bootstrap.BootstrapConfig(
        n_bootstrap=1000,         # More bootstrap samples for stability.
        sample_size=df.shape[0],
        random_state=42,
        weights=None,
        analysis_funcs=[bootstrap.analysis_logistic_regression]
    )
    input_data = BootstrapDataExample(df=df, bootstrap_config=cfg)
    results = bootstrap.run_bootstrap(input_data)
    
    # Compute the mean estimates across bootstrap samples.
    coef0_mean = results["coeff_0"].mean()
    coef1_mean = results["coeff_1"].mean()
    threshold_boot = -coef0_mean / coef1_mean
    
    # Rather than using the true parameters, we compare against the full-sample regularized estimates.
    tol = 0.2  # Tolerance for regularized estimates.
    # Helpful comments:
    # - For regularized regression the estimator is biased (shrunken) and the asymptotic distribution centers
    #   on the full-sample regularized estimate rather than the true generating parameter.
    assert abs(coef0_mean - full_coef0) < tol, (
        f"Regularized intercept bootstrap mean {coef0_mean} not within {tol} of full-sample {full_coef0}"
    )
    assert abs(coef1_mean - full_coef1) < tol, (
        f"Regularized coefficient bootstrap mean {coef1_mean} not within {tol} of full-sample {full_coef1}"
    )
    # Optionally, verify the threshold consistency.
    full_threshold = -full_coef0 / full_coef1
    assert abs(threshold_boot - full_threshold) < tol, (
        f"Bootstrap threshold {threshold_boot} not within {tol} of full-sample threshold {full_threshold}"
    )
    # TODO: The regularization implementation in stastmodels is either poor
    # or I've failed to call it correctly. Currently, this perfect example for
    # it provides coefficients of exactly 0. Investigate why this happens.
    # If no cause is found use scikit learn for regularization instead.
    # I've looked into it and I seem to be calling it correctly. Time
    # to switch to scikit-learn for regularization.


# Note: The procedural tests for logistic regression are expensive as even
# somewhat reliable estimates require a lot of samples.
# Note: In practise, setting n_bootstrap very high can help for smaller
# sample sizes, but this is not a good test for the code.
# Skip this test for now
# @pytest.mark.skip(reason="Skipping expensive bootstrap regression test for now")
@settings(deadline=None, max_examples=2)
@given(logistic_data=logistic_data_strategy())
def test_bootstrap_logistic_regression_estimates(logistic_data):
    df, true_beta0, true_beta1 = logistic_data
    log_reg_config = LogRegConfig(engine="statsmodels", regularize=False)
    stats_fn = lambda indices, data: bootstrap.analysis_logistic_regression(
        indices=indices,
        data=data,
        x_cols=["col_1"],
        y_col="col_2",
        config=log_reg_config
    )
    cfg = bootstrap.BootstrapConfig(
        n_bootstrap=100,
        sample_size=len(df),
        random_state=123,
        weights=None,
        analysis_funcs=[stats_fn]
    )
    input_data = BootstrapDataExample(df=df, bootstrap_config=cfg)
    results = bootstrap.run_bootstrap(input_data)
        
    # The coefficients computed across bootstrap samples.
    col0 = results["coeff_0"].mean()  
    col1 = results["coeff_1"].mean()
    threshold_boot = -col0 / col1
    
    # Compute the full-sample estimate.

    config_full = log_reg_config
    X_full = df["col_1"].values.reshape(-1, 1)
    y_full = df["col_2"].values
    full_res = fit_logistic(X_full, y_full, config_full)
    full_coef0, full_coef1 = full_res.coeffs
    full_threshold = -full_coef0 / full_coef1
    
    tol = 0.5  # Tolerance for bootstrap versus full-sample estimates.
    assert abs(col0 - full_coef0) < tol, f"Intercept estimate {col0} not within {tol} of full-sample {full_coef0}"
    assert abs(col1 - full_coef1) < tol, f"Coefficient estimate {col1} not within {tol} of full-sample {full_coef1}"
    assert abs(threshold_boot - full_threshold) < tol, (
        f"Bootstrap threshold {threshold_boot} not within {tol} of full-sample threshold {full_threshold}"
    )
    
    # Compare full-sample estimate with true parameters.
    tol = 1  # Tolerance to account for finite-sample variability.
    assert abs(col0 - true_beta0) < tol, f"Intercept estimate {col0} not within {tol} of true {true_beta0}"
    assert abs(col1 - true_beta1) < tol, f"Coefficient estimate {col1} not within {tol} of true {true_beta1}"
    if np.abs(true_beta1) > 1e-4:
        true_threshold = -true_beta0 / true_beta1
        assert abs(threshold_boot - true_threshold) < tol, f"Threshold {threshold_boot} not within {tol} of true {true_threshold}"


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

# # Simulate a dataset
# np.random.seed(0)
# n_samples = 2000
# X_sim = np.random.normal(0, 1, n_samples)
# # True logistic model: logit(p) = -0.2 + 1.5 * X_sim
# logits = -0.2 + 1.5 * X_sim
# p = 1 / (1 + np.exp(-logits))
# y_sim = np.random.binomial(1, p, n_samples)
# df_sim = pd.DataFrame({'X': X_sim, 'y': y_sim})

# # Create configuration objects
# logreg_config = LogRegConfig(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
# boot_config = BootstrapConfig(n_boot=5000, random_state=42)

# # Compute the threshold on the full sample
# threshold_full = compute_threshold(df_sim['X'].values, df_sim['y'].values, logreg_config)
# print("Threshold estimate for full sample:", threshold_full)

# # Bootstrap the threshold estimates
# boot_thresh, conv_flags = bootstrap_threshold(
#     data=df_sim,
#     predictor='X',
#     outcome='y',
#     boot_config=boot_config,
#     logreg_config=logreg_config
# )

# print("First 10 bootstrap threshold estimates:", boot_thresh[:10])
# valid_flags = conv_flags[~np.isnan(boot_thresh)]
# print("Fraction of non-converged bootstrap samples:", np.mean(~valid_flags))
# valid_thresh = boot_thresh[~np.isnan(boot_thresh)]
# print("Mean bootstrap threshold estimate:", np.mean(valid_thresh))
# print("Standard error:", np.std(valid_thresh, ddof=1))

# threshold = boot_thresh[~np.isnan(boot_thresh)].mean()
# true_threshold = -1 * -0.2 / 1.5

# tol = 0.01 # tolerance for the estimates
# assert abs(threshold - true_threshold) < tol, (
#     f"Threshold estimate {threshold} not within {tol} of true {true_threshold}"
# )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
