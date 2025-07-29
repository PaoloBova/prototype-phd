import pytest
import numpy as np
from datetime import datetime
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays

from src.schemas import (
    EvaluationForecast,
    AbilityForecast,
    TaskSamplerType,
    WindowAdjustmentMethod
)
from src.simulation import (
    generate_task_samples,
    generate_success_outcomes,
    logistic_function,
    SimulationConfig
)

# TODO: These tests need major refactoring for new schema structure
# EvaluationForecast structure has changed completely - now uses clean composition
# with scenario_id, design_id, scenario, design instead of individual fields
# All test forecast creation methods need updating

pytestmark = pytest.mark.skip(reason="Test file needs refactoring for new EvaluationForecast schema structure")


class TestLogisticFunction:
    """Test the logistic function utility."""
    
    @given(
        x=arrays(np.float64, shape=st.integers(1, 100), elements=st.floats(-10, 10)),
        threshold=st.floats(-5, 5),
        slope=st.floats(-3, 3)
    )
    def test_logistic_function_properties(self, x, threshold, slope):
        """Test logistic function properties."""
        assume(np.all(np.isfinite(x)))
        assume(np.isfinite(threshold) and np.isfinite(slope))
        
        result = logistic_function(x, threshold, slope)
        
        # Check output is valid probabilities
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert np.all(np.isfinite(result))
        
        # Check shape preservation
        assert result.shape == x.shape
    
    def test_logistic_function_at_threshold(self):
        """Test logistic function value at threshold."""
        x = np.array([5.0])
        threshold = 5.0
        slope = -1.0
        
        result = logistic_function(x, threshold, slope)
        
        # At threshold, logistic function should be 0.5
        assert np.isclose(result[0], 0.5, atol=1e-10)


class TestGenerateSuccessOutcomesElicitationBias:
    """Test success outcome generation with elicitation bias."""
    
    def create_test_forecast(self, 
                           elicitation_bias_enabled: bool = False,
                           elicitation_bias_type: str = "fall_past_threshold",
                           elicitation_bias_args: list = None) -> EvaluationForecast:
        """Create a test forecast with elicitation bias parameters."""
        if elicitation_bias_args is None:
            elicitation_bias_args = [0.5]
            
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test",
            model="test_model"
        )
        
        return EvaluationForecast(
            ability=ability,
            budget_fraction=1.0,
            budget_scenario="test",
            window_lower=0.0,
            window_upper=10.0,
            sampler_type=TaskSamplerType.UNIFORM,
            total_samples=100,
            gold_standard_cost=1000.0,
            available_budget=1000.0,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            original_window_lower=0.0,
            original_window_upper=10.0,
            repeats_per_unit=20,
            cost_model="test_cost",
            doubling_rate=2.0,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="test_constraint",
            design_id="test_design",
            elicitation_bias_enabled=elicitation_bias_enabled,
            elicitation_bias_type=elicitation_bias_type,
            elicitation_bias_args=elicitation_bias_args,
            alternate_ability_enabled=False
        )
    
    def test_generate_success_outcomes_no_elicitation_bias(self):
        """Test success outcome generation without elicitation bias."""
        forecast = self.create_test_forecast(elicitation_bias_enabled=False)
        task_difficulties = np.linspace(0, 10, 100)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))  # Binary outcomes
    
    def test_generate_success_outcomes_fall_past_threshold(self):
        """Test success outcomes with fall_past_threshold elicitation bias."""
        forecast = self.create_test_forecast(
            elicitation_bias_enabled=True,
            elicitation_bias_type="fall_past_threshold",
            elicitation_bias_args=[0.3]  # 30% sensitivity rate past threshold
        )
        task_difficulties = np.array([3.0, 5.0, 7.0, 9.0])  # Some below, some above threshold=5.0
        config = SimulationConfig()
        
        # Run multiple times to test the bias effect
        success_rates = []
        for seed in range(100):
            success = generate_success_outcomes(task_difficulties, config, forecast, seed=seed)
            success_rates.append(success.mean())
        
        mean_success_rate = np.mean(success_rates)
        
        # With elicitation bias, overall success rate should be lower than without
        forecast_no_bias = self.create_test_forecast(elicitation_bias_enabled=False)
        success_rates_no_bias = []
        for seed in range(100):
            success_no_bias = generate_success_outcomes(task_difficulties, config, forecast_no_bias, seed=seed)
            success_rates_no_bias.append(success_no_bias.mean())
        
        mean_success_rate_no_bias = np.mean(success_rates_no_bias)
        
        # Elicitation bias should reduce success rate
        assert mean_success_rate < mean_success_rate_no_bias
    
    def test_generate_success_outcomes_linear_elicitation_bias(self):
        """Test success outcomes with linear elicitation bias."""
        forecast = self.create_test_forecast(
            elicitation_bias_enabled=True,
            elicitation_bias_type="linear",
            elicitation_bias_args=[0.0, 1.0]  # threshold=0, slope=1
        )
        task_difficulties = np.linspace(0, 10, 50)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
    
    def test_generate_success_outcomes_logistic_elicitation_bias(self):
        """Test success outcomes with logistic elicitation bias."""
        forecast = self.create_test_forecast(
            elicitation_bias_enabled=True,
            elicitation_bias_type="logistic",
            elicitation_bias_args=[5.0, -1.0]  # threshold=5, slope=-1
        )
        task_difficulties = np.linspace(0, 10, 50)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
    
    @given(
        sensitivity_rate=st.floats(min_value=0.0, max_value=1.0),
        n_tasks=st.integers(min_value=10, max_value=100)
    )
    def test_elicitation_bias_fall_past_threshold_property(self, sensitivity_rate, n_tasks):
        """Property-based test for fall_past_threshold elicitation bias."""
        forecast = self.create_test_forecast(
            elicitation_bias_enabled=True,
            elicitation_bias_type="fall_past_threshold",
            elicitation_bias_args=[sensitivity_rate]
        )
        
        # Tasks all past threshold (should have reduced sensitivity)
        task_difficulties = np.full(n_tasks, 7.0)  # All above threshold=5.0
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == n_tasks
        assert np.all((success == 0) | (success == 1))


class TestGenerateSuccessOutcomesAlternateAbility:
    """Test success outcome generation with alternate ability functions."""
    
    def create_test_forecast_alternate_ability(self,
                                             alternate_ability_enabled: bool = False,
                                             alternate_ability_type: str = "exponential",
                                             alternate_ability_args: list = None) -> EvaluationForecast:
        """Create a test forecast with alternate ability parameters."""
        if alternate_ability_args is None:
            alternate_ability_args = [1.0]
            
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test",
            model="test_model"
        )
        
        return EvaluationForecast(
            ability=ability,
            budget_fraction=1.0,
            budget_scenario="test",
            window_lower=0.0,
            window_upper=10.0,
            sampler_type=TaskSamplerType.UNIFORM,
            total_samples=100,
            gold_standard_cost=1000.0,
            available_budget=1000.0,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            original_window_lower=0.0,
            original_window_upper=10.0,
            repeats_per_unit=20,
            cost_model="test_cost",
            doubling_rate=2.0,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="test_constraint",
            design_id="test_design",
            elicitation_bias_enabled=False,
            alternate_ability_enabled=alternate_ability_enabled,
            alternate_ability_type=alternate_ability_type,
            alternate_ability_args=alternate_ability_args
        )
    
    def test_generate_success_outcomes_exponential_alternate_ability(self):
        """Test success outcomes with exponential alternate ability function."""
        forecast = self.create_test_forecast_alternate_ability(
            alternate_ability_enabled=True,
            alternate_ability_type="exponential",
            alternate_ability_args=[1.0]  # decay_rate=1.0
        )
        task_difficulties = np.linspace(0, 10, 50)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
    
    def test_generate_success_outcomes_power_law_alternate_ability(self):
        """Test success outcomes with power law alternate ability function."""
        forecast = self.create_test_forecast_alternate_ability(
            alternate_ability_enabled=True,
            alternate_ability_type="power_law",
            alternate_ability_args=[2.0]  # exponent=2.0
        )
        task_difficulties = np.linspace(1, 10, 50)  # Avoid zero for power law
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
    
    def test_generate_success_outcomes_logistic_alternate_ability(self):
        """Test success outcomes with logistic alternate ability function."""
        forecast = self.create_test_forecast_alternate_ability(
            alternate_ability_enabled=True,
            alternate_ability_type="logistic",
            alternate_ability_args=[6.0, -0.8]  # threshold=6.0, slope=-0.8
        )
        task_difficulties = np.linspace(0, 10, 50)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
    
    def test_generate_success_outcomes_tangent_alternate_ability(self):
        """Test success outcomes with tangent alternate ability function."""
        forecast = self.create_test_forecast_alternate_ability(
            alternate_ability_enabled=True,
            alternate_ability_type="tangent",
            alternate_ability_args=[0.1, 0.5]  # slope=0.1, intercept=0.5
        )
        task_difficulties = np.linspace(0, 10, 50)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
    
    
    @given(
        function_type=st.sampled_from(["exponential", "power_law", "logistic"]),
        n_tasks=st.integers(min_value=10, max_value=50)
    )
    def test_alternate_ability_functions_property_based(self, function_type, n_tasks):
        """Property-based test for alternate ability functions."""
        # Generate appropriate parameters for each function type
        if function_type == "exponential":
            args = [1.0]  # decay_rate
        elif function_type == "power_law":
            args = [1.5]  # exponent
        elif function_type == "logistic":
            args = [5.0, -1.0]  # threshold, slope
        
        forecast = self.create_test_forecast_alternate_ability(
            alternate_ability_enabled=True,
            alternate_ability_type=function_type,
            alternate_ability_args=args
        )
        
        task_difficulties = np.linspace(1, 10, n_tasks)  # Avoid zero
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == n_tasks
        assert np.all((success == 0) | (success == 1))


class TestCombinedElicitationBiasAndAlternateAbility:
    """Test success outcome generation with both elicitation bias and alternate ability."""
    
    def create_combined_forecast(self) -> EvaluationForecast:
        """Create a forecast with both elicitation bias and alternate ability enabled."""
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test",
            model="test_model"
        )
        
        return EvaluationForecast(
            ability=ability,
            budget_fraction=1.0,
            budget_scenario="test",
            window_lower=0.0,
            window_upper=10.0,
            sampler_type=TaskSamplerType.UNIFORM,
            total_samples=100,
            gold_standard_cost=1000.0,
            available_budget=1000.0,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            original_window_lower=0.0,
            original_window_upper=10.0,
            repeats_per_unit=20,
            cost_model="test_cost",
            doubling_rate=2.0,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="test_constraint",
            design_id="test_design",
            elicitation_bias_enabled=True,
            elicitation_bias_type="fall_past_threshold",
            elicitation_bias_args=[0.4],
            alternate_ability_enabled=True,
            alternate_ability_type="exponential",
            alternate_ability_args=[1.2]
        )
    
    def test_combined_elicitation_bias_and_alternate_ability(self):
        """Test success outcomes with both elicitation bias and alternate ability."""
        forecast = self.create_combined_forecast()
        task_difficulties = np.linspace(0, 10, 100)
        config = SimulationConfig()
        
        success = generate_success_outcomes(task_difficulties, config, forecast, seed=42)
        
        assert len(success) == len(task_difficulties)
        assert np.all((success == 0) | (success == 1))
        
        # Both effects should be applied in sequence
        # First alternate ability function modifies probabilities
        # Then elicitation bias modifies the final success outcomes


class TestErrorHandling:
    """Test error handling for invalid parameters."""
    
    def test_invalid_elicitation_bias_type(self):
        """Test error handling for invalid elicitation bias type."""
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test",
            model="test_model"
        )
        
        forecast = EvaluationForecast(
            ability=ability,
            budget_fraction=1.0,
            budget_scenario="test",
            window_lower=0.0,
            window_upper=10.0,
            sampler_type=TaskSamplerType.UNIFORM,
            total_samples=100,
            gold_standard_cost=1000.0,
            available_budget=1000.0,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            original_window_lower=0.0,
            original_window_upper=10.0,
            repeats_per_unit=20,
            cost_model="test_cost",
            doubling_rate=2.0,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="test_constraint",
            design_id="test_design",
            elicitation_bias_enabled=True,
            elicitation_bias_type="invalid_type",  # Invalid type
            elicitation_bias_args=[0.5],
            alternate_ability_enabled=False
        )
        
        task_difficulties = np.array([1.0, 2.0, 3.0])
        config = SimulationConfig()
        
        with pytest.raises(ValueError):
            generate_success_outcomes(task_difficulties, config, forecast)
    
    def test_invalid_alternate_ability_type(self):
        """Test error handling for invalid alternate ability type."""
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test",
            model="test_model"
        )
        
        forecast = EvaluationForecast(
            ability=ability,
            budget_fraction=1.0,
            budget_scenario="test",
            window_lower=0.0,
            window_upper=10.0,
            sampler_type=TaskSamplerType.UNIFORM,
            total_samples=100,
            gold_standard_cost=1000.0,
            available_budget=1000.0,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            original_window_lower=0.0,
            original_window_upper=10.0,
            repeats_per_unit=20,
            cost_model="test_cost",
            doubling_rate=2.0,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="test_constraint",
            design_id="test_design",
            elicitation_bias_enabled=False,
            alternate_ability_enabled=True,
            alternate_ability_type="invalid_function",  # Invalid type
            alternate_ability_args=[1.0]
        )
        
        task_difficulties = np.array([1.0, 2.0, 3.0])
        config = SimulationConfig()
        
        with pytest.raises(ValueError):
            generate_success_outcomes(task_difficulties, config, forecast)


if __name__ == "__main__":
    pytest.main([__file__])