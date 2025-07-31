import numpy as np
import matplotlib.pyplot as plt

def compute_e_star_closed_form(l, u, k, t1, t2, s):
    """
    Analytic solution for the threshold e* such that
        ∫_l^u L(x; t1) dx = ∫_l^u L(x; t2) h_e(x) dx,
    where L(x; t) = 1 / (1 + exp(-k*(x - t))) and
        h_e(x) = 1  if x < e
               = s  if x >= e,  with 0 <= s < 1.

    Parameters
    ----------
    l, u : float
        Interval endpoints (l < u).
    k : float
        Common logistic slope (k > 0).
    t1, t2 : float
        Curve centers, with t2 > t1.
    s : float
        Post-threshold level (0 <= s < 1).

    Returns
    -------
    e_star : float
        The unique root e* in (l, u).

    Raises
    ------
    ValueError
        If parameters violate assumptions or no valid e* exists.
    """
    # -- Parameter validation --
    if not (l < u):
        raise ValueError("Require l < u.")
    if k >= 0:
        raise ValueError("Require k < 0.")
    if not (0 <= s < 1):
        raise ValueError("Require 0 <= s < 1.")
    
    # -- Helper for the logistic antiderivative term Lambda(x; t) = ln(1 + exp(k*(x - t))) --
    def Lambda(x, t):
        return np.log1p(np.exp(k * (x - t)))
    
    # -- Compute area under the first curve and total area under the second curve --
    A1 = (Lambda(u, t1) - Lambda(l, t1)) / k
    B_tot = (Lambda(u, t2) - Lambda(l, t2)) / k
    
    print(f"A1: {A1}, B_tot: {B_tot}")  # Debugging output
    if A1 > B_tot:
        raise ValueError("No valid e*: A1 must be less than or equal to B_tot.")
    
    # -- Compute D = (A1 - s * B_tot) / (1 - s) --
    D = (A1 - s * B_tot) / (1 - s)
    
    # -- Form the argument of the final log in a numerically stable way --
    beta = k * (l - t2)             # exponent shift for lower limit
    alpha = k * D                   # exponent shift from D
    log1p_exp_beta = np.log1p(np.exp(beta))  # log(1 + exp(beta))
    
    log_num = alpha + log1p_exp_beta
    # Here log_num = ln(e^{kD}(1 + e^{k(l - t2)}))
    # We need log_num > 0 so that exp(log_num) > 1 and exp(log_num) - 1 > 0.
    if log_num <= 0:
        raise ValueError("No valid e*: logarithm argument must exceed 1.")
    
    # Compute log(E) = ln(e^{log_num} - 1) stably:
    #    = log_num + ln(1 - e^{-log_num})
    log_E = log_num + np.log1p(-np.exp(-log_num))
    
    # -- Final threshold --
    e_star = t2 + log_E / k
    return e_star

# Example usage:
# e = compute_e_star_closed_form(l=0.0, u=10.0, k=1.0, t1=3.0, t2=7.0, s=0.5)
# print(f"e* = {e:.6f}")

TRAINING_COMPUTE_GROWTH_RATE = 4.6 # 4.6x per year
ABILITY_DOUBLING_RATE = 1 / 0.7 # 1 doubling every 0.7 years
ABILITY_GROWTH_PER_YEAR = 2**ABILITY_DOUBLING_RATE

# 5x training compute growth leads to time ability gain of ??
YEARS_FOR_5X_TRAINING_COMPUTE_GAIN = np.log(5) / np.log(TRAINING_COMPUTE_GROWTH_RATE)
COMPUTE_EQUIVALENT_GAIN = YEARS_FOR_5X_TRAINING_COMPUTE_GAIN * ABILITY_GROWTH_PER_YEAR
CLAUDE_TIME_HORIZON = 59 * 60  # 59 minutes in seconds
AVG_SLOPE = -0.6665
EVALUATION_DATA_LOWEST_DIFFICULTY = 0.26  # Lowest difficulty in evaluation data used for forecast (log2 human seconds)
EVALUATION_DATA_HIGHEST_DIFFICULTY = 16.03  # Highest difficulty in evaluation data used for forecast (log2 human seconds)
SENSITIVITY_RATE_AFTER_THRESHOLD = 0
e_star = compute_e_star_closed_form(
    l=EVALUATION_DATA_LOWEST_DIFFICULTY,
    u=EVALUATION_DATA_HIGHEST_DIFFICULTY,
    k=AVG_SLOPE,
    t1=np.log2(CLAUDE_TIME_HORIZON / COMPUTE_EQUIVALENT_GAIN),
    t2=np.log2(CLAUDE_TIME_HORIZON),
    s=SENSITIVITY_RATE_AFTER_THRESHOLD,
)

print("Constants:")
print(f"Training Compute Growth Rate: {TRAINING_COMPUTE_GROWTH_RATE} x per year")
print(f"Ability Doubling Rate: {ABILITY_DOUBLING_RATE:.2f} doublings per year")
print(f"Ability Growth Per Year: {ABILITY_GROWTH_PER_YEAR:.2f}")
print(f"Years for 5x Training Compute Gain: {YEARS_FOR_5X_TRAINING_COMPUTE_GAIN:.2f} years")
print(f"Compute Equivalent Gain: {COMPUTE_EQUIVALENT_GAIN:.2f}x ability gain")
print(f"Claude Time Horizon: {CLAUDE_TIME_HORIZON} seconds")
print(f"Average Slope: {AVG_SLOPE:.3f}")
print(f"Elicitation Data Range: [{EVALUATION_DATA_LOWEST_DIFFICULTY}, {EVALUATION_DATA_HIGHEST_DIFFICULTY}]")
print(f"Sensitivity Rate After Threshold: {SENSITIVITY_RATE_AFTER_THRESHOLD:.2f}")
print("Elicitation Threshold Parameters:")
print(f"l = {EVALUATION_DATA_LOWEST_DIFFICULTY:.2f}")
print(f"u = {EVALUATION_DATA_HIGHEST_DIFFICULTY:.2f}")
print(f"k = {AVG_SLOPE:.2f}")
print(f"t1 = {np.log2(CLAUDE_TIME_HORIZON / COMPUTE_EQUIVALENT_GAIN):.2f}")
print(f"t2 = {np.log2(CLAUDE_TIME_HORIZON):.2f}")
print(f"s = {SENSITIVITY_RATE_AFTER_THRESHOLD:.2f}")
print(f"e* = {e_star:.6f} (Elicitation threshold)")

# Remember the computed delta = t2 - t1
delta = np.log2(CLAUDE_TIME_HORIZON) - np.log2(CLAUDE_TIME_HORIZON / COMPUTE_EQUIVALENT_GAIN)
print(f"Delta = {delta:.6f} (Difference between t2 and t1)")
# Or use compute equivalent gain as a multiplier instead of delta

# Use the relevant base elicitation threshold in our elicitation bias configs
# For s=0, we should get e* = 11.039387
# For s = 0.25, we should get e* = 10.329913
# For s = 0.5, we should get e* = 9.125054

# Budget calibrations
# The gold standard cost is computed by multiplying marginal costs of added inference
# by the average inference cost of the gold standard evaluation window.
# Simple linear scaling for logistic ability shift when facing resource constraint
# (computation of gold standard cost is not needed here as we already set budget
# constraints as a percentage).
# For threshold filter, we handle budget constraints by adjusting the elicitation
# threshold only as far we can afford the added inference costs. This means
# we need absolute costs so that we can find the the threshold such that
# the added inference costs equal the budget constraint.

def plot_area_comparison(l, u, k, t1, t2, s, e_star):
    """
    Plot the two logistic curves and show how the Heaviside function
    equalizes the areas under the curves.
    """
    x = np.linspace(l, u, 1000)
    
    # Define the logistic curves
    L1 = 1 / (1 + np.exp(-k * (x - t1)))
    L2 = 1 / (1 + np.exp(-k * (x - t2)))
    
    # Define the Heaviside function
    h_e = np.where(x < e_star, 1.0, s)
    
    # Filtered second curve
    L2_filtered = L2 * h_e
    
    plt.figure(figsize=(12, 8))
    
    # Plot the curves
    plt.plot(x, L1, 'b-', linewidth=2, label=f'L(x; t₁={t1:.2f})')
    plt.plot(x, L2, 'r--', linewidth=2, label=f'L(x; t₂={t2:.2f})')
    plt.plot(x, L2_filtered, 'g-', linewidth=2, label=f'L(x; t₂) × h_e(x)')
    
    # Add vertical line at threshold
    plt.axvline(x=e_star, color='orange', linestyle=':', linewidth=2, 
                label=f'Threshold e*={e_star:.3f}')
    
    # Fill areas to show the equality
    plt.fill_between(x, 0, L1, alpha=0.3, color='blue', 
                     label='Area under L(x; t₁)')
    plt.fill_between(x, 0, L2_filtered, alpha=0.3, color='green',
                     label='Area under filtered L(x; t₂)')
    
    # Calculate and display actual areas
    dx = x[1] - x[0]
    area1 = np.trapezoid(L1, dx=dx)
    area2_filtered = np.trapezoid(L2_filtered, dx=dx)
    
    plt.title(f'Elicitation Threshold Visualization\n'
              f'Area₁ = {area1:.4f}, Area₂ (filtered) = {area2_filtered:.4f}')
    plt.xlabel('Difficulty Level')
    plt.ylabel('Success Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add text box with parameters
    textstr = f'Parameters:\nk = {k:.3f}\nt₁ = {t1:.3f}\nt₂ = {t2:.3f}\ns = {s:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.show()
    
    return area1, area2_filtered

# Create the visualization
print("\nGenerating area comparison plot...")
area1, area2_filtered = plot_area_comparison(
    l=EVALUATION_DATA_LOWEST_DIFFICULTY,
    u=EVALUATION_DATA_HIGHEST_DIFFICULTY,
    k=AVG_SLOPE,
    t1=np.log2(CLAUDE_TIME_HORIZON / COMPUTE_EQUIVALENT_GAIN),
    t2=np.log2(CLAUDE_TIME_HORIZON),
    s=SENSITIVITY_RATE_AFTER_THRESHOLD,
    e_star=e_star
)

print(f"Numerical verification:")
print(f"Area under L(x; t₁): {area1:.6f}")
print(f"Area under filtered L(x; t₂): {area2_filtered:.6f}")
print(f"Difference: {abs(area1 - area2_filtered):.6f}")

