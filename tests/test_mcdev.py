import numpy as np
import pytest
from hypothesis import given, strategies as st
from prototype_phd.methods.mcdev import compute_optimal_demands, DemandConfig
import prototype_phd.methods.mcdev as mcdev

@given(
    B=st.floats(min_value=0.1, max_value=1e3),
    alpha=st.sampled_from([0.0, 0.5, 0.9]),
    K=st.integers(min_value=5, max_value=30),
)
def test_iterative_solver(B, alpha, K):
    """
    Property-based test:
     - Generate valid prices with psi and gamma as ones.
     - Check returned demands are non-negative and spending does not exceed budget.
    """
    p = np.random.uniform(low=0.1, high=10.0, size=K)
    psi = np.ones(K)
    gamma = np.ones(K)
    
    scenario_config = mcdev.ScenarioConfig(
            scenario_name="Exponential Increase",
            scenario_func=mcdev.scenario_exponential,
            K=K)

    config = DemandConfig(scenario_config=scenario_config,
                          B=B, p=p, psi=psi, gamma=gamma, alpha=alpha)
    x_star = compute_optimal_demands(config)

    assert np.all(x_star >= -1e-7), "Demands must be >= 0"
    total_spending = np.dot(p, x_star)
    assert total_spending <= B + 1e-5, "Total spending should not exceed budget significantly."

if __name__ == "__main__":
    pytest.main()
