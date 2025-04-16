import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model

statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF(True)

from prototype_phd.stats import (
    fit_logistic,
    compute_threshold_from_result,
    run_diagnostics,
    compare_link_functions,
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

if __name__ == "__main__":
    pytest.main([__file__])