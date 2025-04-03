import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model

statsmodels.genmod.generalized_linear_model.SET_USE_BIC_LLF(True)

from prototype_phd.stats import (
    robust_logistic_fit,
    compute_threshold,
    run_diagnostics,
    compare_link_functions
)

def test_binary_data():
    """
    Test robust_logistic_fit with one predictor and binary outcomes.
    Expect the estimated 50% threshold (i.e. -beta0/beta1) to be near 5.
    """
    # Create synthetic data: predictor x, binary outcome y
    x = np.linspace(0, 10, 100)
    beta0, beta1 = -5, 1  # so threshold is approximately 5
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
    np.random.seed(0)
    y = np.random.binomial(1, p)

    # Fit without frequency weights (each row is a single trial)
    result = robust_logistic_fit(x, y, freq_weights=None, regularize=False,
                                 link=sm.families.links.logit())
    threshold = compute_threshold(result)
    # Check that threshold is computed and close to expected value.
    assert threshold is not None
    assert abs(threshold - 5) < 1.0


def test_proportion_data():
    """
    Test robust_logistic_fit with one predictor and proportion outcomes.
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

    result = robust_logistic_fit(x, y, freq_weights=trials, regularize=False,
                                 link=sm.families.links.logit())
    threshold = compute_threshold(result)
    assert threshold is not None
    assert abs(threshold - 5) < 1.0


def test_input_validation():
    """
    Test that a length mismatch between x and y raises a ValueError.
    """
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 1, 40)  # Mismatch length
    with pytest.raises(ValueError):
        robust_logistic_fit(x, y)

def test_run_diagnostics():
    """
    Test that run_diagnostics returns a dictionary with expected keys.
    """
    x = np.linspace(0, 10, 50)
    beta0, beta1 = -5, 1
    p = 1 / (1 + np.exp(-(beta0 + beta1 * x)))
    np.random.seed(2)
    y = np.random.binomial(1, p)
    result = robust_logistic_fit(x, y, link=sm.families.links.logit())
    diagnostics = run_diagnostics(result)
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

    metrics, best_link = compare_link_functions(x, y, freq_weights=trials)
    # Expected keys in metrics.
    for link in ["logit", "probit", "cloglog"]:
        assert link in metrics
    assert best_link in metrics
