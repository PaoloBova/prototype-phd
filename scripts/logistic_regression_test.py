import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Simulate data for demonstration (data should be in (0,1))
np.random.seed(42)
t_data = np.linspace(0, 10, 100)
# True parameters for logistic function with L fixed at 1
k_true, t0_true = 1.0, 5.0

# Generate "true" probabilities from the logistic function
def logistic(t, k, t0):
    return 1 / (1 + np.exp(-k * (t - t0)))

y_true = logistic(t_data, k_true, t0_true)

# Simulate observed binary outcomes or proportions.
# For proportions, you might have an effective sample size (n)
# Here, we'll assume n=30 for each observation and simulate proportions.
n = 30
y_counts = np.random.binomial(n, y_true)
y_data = y_counts / n

# Ensure the data are strictly between 0 and 1 (adjust if necessary)
epsilon = 1e-4
y_data = np.clip(y_data, epsilon, 1-epsilon)

# Construct the design matrix
X = sm.add_constant(t_data)

# Fit a GLM with a binomial family and a logit link.
# For proportion data, you need to pass the counts and the total trials.
model = sm.GLM(y_data, X, family=sm.families.Binomial(), freq_weights=np.full_like(y_data, n))
results = model.fit()

print(results.summary())

# The model estimates coefficients for:
# logit(y) = beta0 + beta1 * t
# To relate this back to the logistic model parameters:
#   beta1 corresponds to k, and beta0 = -k*t0.
k_est = results.params[1]
t0_est = -results.params[0] / k_est

print("\nEstimated Parameters:")
print("Estimated k:", k_est)
print("Estimated t0:", t0_est)

# Plotting the fitted curve against the observed proportions
t_fit = np.linspace(min(t_data), max(t_data), 200)
X_fit = sm.add_constant(t_fit)
logit_fit = results.predict(X_fit)
y_fit = 1 / (1 + np.exp(-np.log((1-logit_fit)/logit_fit)))  # Alternatively, just use model.predict

# Since we're predicting on the probability scale directly:
y_fit = results.predict(X_fit)

plt.figure(figsize=(8, 5))
plt.scatter(t_data, y_data, label='Observed Proportions', color='black')
plt.plot(t_fit, y_fit, label='Fitted Logistic Curve (GLM)', color='red')
plt.xlabel('t')
plt.ylabel('Proportion')
plt.title('GLM Fit for Logistic Model')
plt.legend()
plt.show()
