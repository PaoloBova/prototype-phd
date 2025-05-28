# Simulation Approaches for Evaluation Forecasting

This document describes the different simulation approaches implemented for forecasting model evaluation performance and analyzing estimation reliability under different conditions.

## Overview

When forecasting how well we can measure model capabilities with limited evaluation resources, there are several sources of uncertainty to consider:

1. **Task Selection Uncertainty**: Which specific tasks are included in an evaluation
2. **Success/Failure Uncertainty**: Stochasticity in model performance on tasks
3. **Task Correlation**: The degree to which models succeed on the same tasks

Our simulation framework supports multiple approaches to model these uncertainties.

## Simulation Methods

### Bootstrap Sampling

Bootstrap sampling involves creating a fixed set of task-outcome pairs, then resampling from this set with replacement to create each simulation sample. This approach:

- Maintains the joint distribution of tasks and outcomes
- Models the uncertainty from having a finite evaluation set
- Does not introduce new task difficulties or outcomes
- Is useful when we want to understand the variability from sampling alone

### Monte Carlo Sampling

Monte Carlo sampling generates entirely new tasks and outcomes for each simulation replicate. This approach:

- Models both task selection and success/failure uncertainty
- Can better represent the super-population of potential tasks
- Allows different sources of randomness in each sample
- Is more appropriate when we want to understand the full range of possible evaluation outcomes

### Correlation Models

We provide several options for modeling correlation between task successes:

#### Independent Model (None)

- Success on each task is independent
- Success probability follows a logistic curve based on task difficulty
- Different models or runs have uncorrelated patterns of successes and failures
- Tasks at the same difficulty have the same probability of success, but random outcome

#### Fixed Order Model

- Tasks have a fixed order in which models learn to solve them
- Different models/runs will show identical patterns of success and failure
- Represents the extreme case where task successes are perfectly correlated

#### Mixture Model

- Combines aspects of both independent and fixed order models
- Correlation strength parameter controls the balance:
  - 0 = fully independent (standard logistic model)
  - 1 = fully correlated (fixed order model)
- Can model partial correlations, where similar models tend to succeed on similar tasks

## Implementation Details

The simulation is parameterized by a `SimulationConfig` object with these fields:

- `method`: Which simulation method to use ('bootstrap' or 'monte_carlo')
- `n_samples`: Number of simulation samples to generate
- `sample_size`: Size of each sample (if None, uses original data size)
- `random_seed`: Random seed for reproducibility
- `correlation_model`: Model for task success correlation
- `correlation_strength`: Strength of correlation (0-1)

For each simulation, we:

1. Generate task difficulties based on the sampling distribution and window
2. Generate success/failure outcomes based on the logistic model and correlation settings
3. Apply the specified analysis function (e.g., threshold estimator)
4. Calculate summary statistics across all simulation samples

## Choosing the Right Approach

Consider these factors when selecting a simulation approach:

- **Bootstrap**: Use when you want to understand the variability from having a finite evaluation set
- **Monte Carlo**: Use when you want to model the full range of possible evaluation outcomes
- **Independent Model**: Use when you believe model success on tasks is independent
- **Fixed Order**: Use when you believe models learn to solve tasks in a consistent order
- **Mixture**: Use when you want to model partial correlation between task successes

The best approach depends on the specific forecasting question and domain knowledge about how models improve over time.
