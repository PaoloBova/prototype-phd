import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model

statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF(True)

import prototype_phd.stats as stats
from prototype_phd.stats import (
    fit_logistic,
    compute_threshold_from_result,
    run_diagnostics,
    compare_link_functions,
    prepare_binned_data,
    LogRegConfig
)

def test_binary_data():
    """
    Test fit_logistic with one predictor and binary outcomes.
    Expect the estimated 50% threshold (i.e. -beta0/beta1) to be near 5.
    """
    # Create synthetic data: predictor x, binary outcome y
    x = np.linspace(0, 10, 100)
    beta0, beta1 = -5, 1  # so threshold is approximately 5
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
    np.random.seed(0)
    y = np.random.binomial(1, p)

    config = LogRegConfig(engine="statsmodels", link=sm.families.links.logit(), regularize=False, freq_weights=None)
    model_result = fit_logistic(x, y, config)
    threshold = compute_threshold_from_result(model_result)
    # Check that threshold is computed and close to expected value.
    assert threshold is not None
    assert abs(threshold - 5) < 1.0


def test_proportion_data():
    """
    Test fit_logistic with one predictor and proportion outcomes.
    Frequency weights (number of trials) are provided.
    The estimated threshold should be close to the true value.
    """
    x = np.linspace(0, 10, 100)
    beta0, beta1 = -5, 1  # true threshold ~5
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
    # Create synthetic trial counts (varying number of trials)
    trials = np.random.randint(20, 40, size=x.shape[0])
    np.random.seed(1)
    counts = np.random.binomial(trials, p)
    y = counts / trials

    config = LogRegConfig(engine="statsmodels", link=sm.families.links.logit(), regularize=False, freq_weights=trials)
    model_result = fit_logistic(x, y, config)
    threshold = compute_threshold_from_result(model_result)
    assert threshold is not None
    assert abs(threshold - 5) < 1.0


def test_input_validation():
    """
    Test that a length mismatch between x and y raises a ValueError.
    """
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 1, 40)  # Mismatch length
    with pytest.raises(ValueError):
        config = LogRegConfig(engine="statsmodels", link=sm.families.links.logit())
        fit_logistic(x, y, config)

def test_run_diagnostics():
    """
    Test that run_diagnostics returns a dictionary with expected keys.
    """
    x = np.linspace(0, 10, 50)
    beta0, beta1 = -5, 1
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
    np.random.seed(2)
    y = np.random.binomial(1, p)
    config = LogRegConfig(engine="statsmodels", link=sm.families.links.logit())
    model_result = fit_logistic(x, y, config)
    diagnostics = run_diagnostics(model_result)
    expected_keys = ["AIC", "BIC", "Deviance", "DF_Resid", "Deviance/DF",
                     "Pearson_Chi2", "Pearson/DF", "Max_Leverage", "Max_Cooks_D"]
    for key in expected_keys:
        assert key in diagnostics

def test_compare_link_functions():
    """
    Test the compare_link_functions utility. It should return a dictionary of
    AIC/BIC metrics for each link and a best link among 'logit', 'probit', and 'cloglog'.
    """
    x = np.linspace(0, 10, 50)
    beta0, beta1 = -5, 1
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
    # Use constant trial count for simplicity
    trials = np.full_like(x, 30, dtype=int)
    np.random.seed(3)
    counts = np.random.binomial(trials, p)
    y = counts / trials
    config = LogRegConfig(engine="statsmodels", freq_weights=trials)
    metrics, best_link = compare_link_functions(x, y, config=config)
    # Expected keys in metrics.
    for link in ["logit", "probit", "cloglog"]:
        assert link in metrics
    assert best_link in metrics

def test_all_ones():
    """
    Test fit_logistic when y is all ones.
    Expect model_result warning 'all_success' and intercept set to -1.
    """
    x = np.linspace(1, 10, 100)
    y = np.ones(100)
    config = LogRegConfig(engine="statsmodels", link=sm.families.links.logit())
    model_result = fit_logistic(x, y, config)
    assert model_result.warning == "all_success"
    assert model_result.coeffs[0] == -1

def test_all_zeros():
    """
    Test fit_logistic when y is all zeros.
    Expect model_result warning 'all_failure' and intercept set to -1.
    """
    x = np.linspace(1, 10, 100)
    y = np.zeros(100)
    config = LogRegConfig(engine="statsmodels", link=sm.families.links.logit())
    model_result = fit_logistic(x, y, config)
    assert model_result.warning == "all_failure"
    assert model_result.coeffs[0] == -1

def test_scikit_learn_binned_data():
    """
    Bin continuous logistic data into proportions, expand into
    (x, y=1) and (x, y=0) rows with sample_weight = [#success, #failure].
    Then fit with scikit‑learn engine and check threshold ≈ -beta0/beta1.
    """
    # simulate raw data
    np.random.seed(0)
    n_raw = 10000
    beta0, beta1 = -2.0, 0.5
    # beta0, beta1 = 0, -3.0
    x_raw = np.random.uniform(-10, 20, size=n_raw)
    # x_raw = np.linspace(-5, 10, num=n_raw)
    logits = beta0 + beta1 * x_raw
    p_raw = 1 / (1 + np.exp(-logits))
    y_raw = np.random.binomial(1, p_raw)
    raw_df = pd.DataFrame({'X': x_raw, 'Y': y_raw})

    # prepare binned data (use more bins for tighter fit)
    df_exp = prepare_binned_data(raw_df, 'X', 'Y', n_bins=100)
    X = df_exp['x'].values
    y = df_exp['y'].values
    sw = df_exp['sample_weight'].values

    config = LogRegConfig(engine="scikit-learn",
                          solver="lbfgs",
                          C=1,
                          sample_weight=sw)
    # config = LogRegConfig(engine="statsmodels", freq_weights=sw)
    result = fit_logistic(X, y, config)
    threshold = compute_threshold_from_result(result)
    expected = -beta0 / beta1

    # Plot the binned data
    import matplotlib.pyplot as plt
    totals_by_x = df_exp.groupby('x')['sample_weight'].sum()
    rates = df_exp[df_exp["y"] == 1]["sample_weight"].values / totals_by_x
    xs = df_exp[df_exp["y"] == 1]["x"].values
    plt.scatter(xs, rates, label='Binned Data')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.title('Binned Data for Logistic Regression')
    # Plot fitted logistic curve accounting for scaler
    X_plot = np.linspace(X.min(), X.max(), 100)
    coeffs = result.coeffs
    if result.scaler is not None:
        z = result.scaler.transform(X_plot.reshape(-1, 1)).flatten()
    else:
        z = X_plot
    logits_plot = coeffs[0] + coeffs[1] * z
    p_plot = 1 / (1 + np.exp(-logits_plot))
    plt.plot(X_plot, p_plot, 'r-', label='Fitted Logistic Curve')
    # Plot threshold line
    plt.axvline(threshold, color='g', linestyle='--', label='Threshold')
    # Plot expected threshold line
    plt.axvline(expected, color='b', linestyle=':', label='Expected Threshold')
    
    plt.legend()
    plt.show()
    
    # with 100 bins, should converge within 0.2
    assert result.convergence
    assert abs(threshold - expected) < 0.2

if __name__ == "__main__":
    pytest.main([__file__])