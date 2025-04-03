import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.genmod.families.links as sm_links
import logging

import prototype_phd.data_utils as data_utils
from prototype_phd.stats import (
    robust_logistic_fit,
    compute_threshold,
    run_diagnostics,
    compare_link_functions
)

data_utils.setup_logging(log_path='logs/stats.log')

# Generate synthetic data.
t = np.linspace(0, 10, 100)

def logistic(t, k, t0):
    return 1 / (1 + np.exp(-k * (t - t0)))

true_probs = logistic(t, 1, 5)

# Simulate data for proportion outcomes with varying trial counts.
trials = np.random.randint(20, 40, size=t.shape[0])
y_counts = np.random.binomial(trials, true_probs)
y = y_counts / trials

# Fit model using robust_logistic_fit.
result = robust_logistic_fit(t, y, freq_weights=trials,
                                regularize=False, link=sm_links.Logit())
logging.info("Fitted model summary:\n%s", result.summary())

# Run diagnostics.
diagnostics = run_diagnostics(result)

# Compare link functions.
metrics, best_link = compare_link_functions(t, y, freq_weights=trials)
logging.info("Link function metrics: %s", metrics)
logging.info("Suggested best link: %s", best_link)

# Plot observed proportions and fitted logistic curve.
plt.scatter(t, y, label='Observed proportions')
X_plot = sm.add_constant(pd.DataFrame(t, columns=['x']))
fitted_probs = result.predict(X_plot)
plt.plot(t, fitted_probs, 'r-', label='Fitted logistic curve')
plt.xlabel('Predictor (t)')
plt.ylabel('Proportion')
plt.legend()

# Generate synthetic data.
t = np.linspace(0, 10, 100)
def logistic(t, k, t0):
    return 1 / (1 + np.exp(-k * (t - t0)))
true_probs = logistic(t, 1, 5)

# For demonstration, use constant trial count.
trials = np.full_like(t, 30, dtype=int)
np.random.seed(42)
y_counts = np.random.binomial(trials, true_probs)
y = y_counts / trials

# Fit the logistic regression model.
result = robust_logistic_fit(t, y, freq_weights=trials, regularize=False,
                                link=sm.families.links.Logit())

# Extract coefficient estimates.
beta0, beta1 = result.params.iloc[0], result.params.iloc[1]
threshold = compute_threshold(result)

# Compute standard error of the threshold using the delta method.
cov = result.cov_params()
var_threshold = (1/beta1**2) * cov.iloc[0, 0] + (beta0**2/(beta1**4)) * cov.iloc[1, 1] - \
                (2 * beta0/(beta1**3)) * cov.iloc[0, 1]
se_threshold = np.sqrt(var_threshold)

# Compute the 95% confidence interval for the threshold.
ci_lower = threshold - 1.96 * se_threshold
ci_upper = threshold + 1.96 * se_threshold

# Plot data and fitted curve.
plt.scatter(t, y, label='Observed proportions', color='blue', alpha=0.6)
t_fit = np.linspace(t.min(), t.max(), 200)
X_fit = sm.add_constant(pd.DataFrame(t_fit, columns=['x']))
fitted_probs = result.predict(X_fit)
plt.plot(t_fit, fitted_probs, 'r-', label='Fitted logistic curve')

# Add vertical line for threshold and a shaded CI region.
plt.axvline(threshold, color='green', linestyle='--', label='Threshold')
plt.fill_betweenx([0, 1], ci_lower, ci_upper, color='green', alpha=0.2,
                label='95% CI for threshold')

plt.xlabel('Predictor (t)')
plt.ylabel('Proportion')
plt.legend()
plt.title(f"Threshold = {threshold:.2f} (SE = {se_threshold:.2f})")
plt.show()

# For a given new set of x values:
x_new = np.linspace(0, 10, 200)
X_new = sm.add_constant(np.column_stack([x_new]))
# Predicted linear predictors and probabilities
eta = result.predict(X_new, which="linear")

p = 1 / (1 + np.exp(-eta))  # logistic function

# Compute standard error for each linear predictor:
cov_beta = result.cov_params().values
SE_eta = np.array([np.sqrt(np.dot(np.dot(np.array([1, xi]).T, cov_beta), 
                                  np.array([1, xi]))) for xi in x_new])

# Apply delta method to get SE on probability scale:
SE_p = p * (1 - p) * SE_eta

# Compute 95% confidence intervals:
upper = p + 1.96 * SE_p
lower = p - 1.96 * SE_p

# Plot the results
plt.scatter(result.model.data.orig_exog.iloc[:,1], result.model.endog,
            label='Observed Data', alpha=0.5)
plt.plot(x_new, p, 'r-', label='Fitted Logistic Curve')
plt.fill_between(x_new, lower, upper, color='gray', alpha=0.3,
                 label='95% Confidence Band')
plt.xlabel('x')
plt.ylabel('Predicted Probability')
plt.legend()
plt.show()
