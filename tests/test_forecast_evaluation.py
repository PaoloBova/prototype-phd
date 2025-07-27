import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.schemas import (
    AbilityForecast, 
    EvaluationScenario, 
    EvaluationForecast,
    EvaluationDesign,
    ElicitationBiasConfig,  # Will fail until implemented
    AlternateAbilityConfig,  # Will fail until implemented
    TaskSamplerType,
    WindowAdjustmentMethod
)
from src.forecast_evaluation import (
    EvaluationConfig,
    define_elicitation_bias,
    define_alternate_ability_params,  # Will fail until implemented
    calculate_evaluation_forecast
)


class TestElicitationBiasConfig:
    """Test the ElicitationBiasConfig Pydantic model."""
    
    def test_elicitation_bias_config_inline_dict(self):
        """Test creating ElicitationBiasConfig with inline dictionary source."""
        config = ElicitationBiasConfig(
            source={"type": "fall_past_threshold", "args": [0.5]},
            enabled=True
        )
        assert config.enabled is True
        assert isinstance(config.source, dict)
        assert config.source["type"] == "fall_past_threshold"
        assert config.source["args"] == [0.5]
    
    def test_elicitation_bias_config_file_path(self):
        """Test creating ElicitationBiasConfig with file path source."""
        config = ElicitationBiasConfig(
            source="/path/to/elicitation_bias.json",
            enabled=True
        )
        assert config.enabled is True
        assert isinstance(config.source, str)
        assert config.source == "/path/to/elicitation_bias.json"
    
    def test_elicitation_bias_config_defaults(self):
        """Test ElicitationBiasConfig with default values."""
        config = ElicitationBiasConfig()
        assert config.enabled is True
        assert isinstance(config.source, dict)
        assert config.source["type"] == "fall_past_threshold"
    
    @given(
        enabled=st.booleans(),
        bias_type=st.sampled_from(["fall_past_threshold", "linear", "logistic"]),
        args=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=3)
    )
    def test_elicitation_bias_config_property_based(self, enabled, bias_type, args):
        """Property-based test for ElicitationBiasConfig."""
        config = ElicitationBiasConfig(
            source={"type": bias_type, "args": args},
            enabled=enabled
        )
        assert config.enabled == enabled
        assert config.source["type"] == bias_type
        assert config.source["args"] == args


class TestAlternateAbilityConfig:
    """Test the AlternateAbilityConfig Pydantic model."""
    
    def test_alternate_ability_config_disabled(self):
        """Test creating AlternateAbilityConfig with disabled state."""
        config = AlternateAbilityConfig(enabled=False)
        assert config.enabled is False
        assert config.function_type is None
        assert config.parameters is None
    
    def test_alternate_ability_config_exponential(self):
        """Test AlternateAbilityConfig with exponential function."""
        config = AlternateAbilityConfig(
            enabled=True,
            function_type="exponential",
            parameters=[1.5],
            description="Exponential decay robustness check"
        )
        assert config.enabled is True
        assert config.function_type == "exponential"
        assert config.parameters == [1.5]
        assert "Exponential" in config.description
    
    def test_alternate_ability_config_power_law(self):
        """Test AlternateAbilityConfig with power law function."""
        config = AlternateAbilityConfig(
            enabled=True,
            function_type="power_law",
            parameters=[2.0]
        )
        assert config.enabled is True
        assert config.function_type == "power_law"
        assert config.parameters == [2.0]
    
    @given(
        enabled=st.booleans(),
        function_type=st.sampled_from(["exponential", "power_law", "tangent", "logistic"]),
        parameters=st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=1, max_size=4)
    )
    def test_alternate_ability_config_property_based(self, enabled, function_type, parameters):
        """Property-based test for AlternateAbilityConfig."""
        config = AlternateAbilityConfig(
            enabled=enabled,
            function_type=function_type if enabled else None,
            parameters=parameters if enabled else None
        )
        assert config.enabled == enabled
        if enabled:
            assert config.function_type == function_type
            assert config.parameters == parameters


class TestEvaluationConfigExtended:
    """Test the extended EvaluationConfig with bias and ability configurations."""
    
    def test_evaluation_config_with_bias_and_ability(self):
        """Test EvaluationConfig includes elicitation bias and alternate ability configs."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source={"type": "linear", "args": [0.0, 1.0]},
                enabled=True
            ),
            alternate_ability=AlternateAbilityConfig(
                enabled=True,
                function_type="power_law",
                parameters=[1.5]
            )
        )
        
        assert hasattr(config, 'elicitation_bias')
        assert hasattr(config, 'alternate_ability')
        assert config.elicitation_bias.enabled is True
        assert config.alternate_ability.enabled is True
        assert config.alternate_ability.function_type == "power_law"


class TestDefineElicitationBias:
    """Test the define_elicitation_bias function."""
    
    def create_test_scenario(self, budget_fraction: float = 0.5) -> EvaluationScenario:
        """Create a test evaluation scenario."""
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test_scenario",
            model="test_model"
        )
        
        return EvaluationScenario(
            ability=ability,
            doubling_rate=2.0,
            budget_fraction=budget_fraction,
            scenario_id="test_scenario_50pct",
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="static_50pct",
            cost_model="test_cost_model"
        )
    
    def test_define_elicitation_bias_inline_config(self):
        """Test define_elicitation_bias with inline configuration."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source={"type": "fall_past_threshold", "args": [0.5]},
                enabled=True
            )
        )
        
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(config, scenario)
        
        assert isinstance(result, dict)
        assert result["elicitation_bias_enabled"] is True
        assert result["elicitation_bias_type"] == "fall_past_threshold"
        assert result["elicitation_bias_args"] == [0.5]
    
    def test_define_elicitation_bias_disabled(self):
        """Test define_elicitation_bias when disabled."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(enabled=False)
        )
        
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(config, scenario)
        
        assert isinstance(result, dict)
        assert result["elicitation_bias_enabled"] is False
    
    def test_define_elicitation_bias_file_source(self):
        """Test define_elicitation_bias with file path source."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source="/path/to/bias_config.json",
                enabled=True
            )
        )
        
        # This should attempt to load from file and fall back gracefully
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(config, scenario)
        assert isinstance(result, dict)
        # Should have some form of error handling or fallback


class TestDefineAlternateAbilityParams:
    """Test the define_alternate_ability_params function."""
    
    def test_define_alternate_ability_params_disabled(self):
        """Test define_alternate_ability_params when disabled."""
        config = EvaluationConfig(
            alternate_ability=AlternateAbilityConfig(enabled=False)
        )
        
        result = define_alternate_ability_params(config)
        
        assert isinstance(result, dict)
        assert result["alternate_ability_enabled"] is False
    
    def test_define_alternate_ability_params_exponential(self):
        """Test define_alternate_ability_params with exponential function."""
        config = EvaluationConfig(
            alternate_ability=AlternateAbilityConfig(
                enabled=True,
                function_type="exponential",
                parameters=[1.5]
            )
        )
        
        result = define_alternate_ability_params(config)
        
        assert isinstance(result, dict)
        assert result["alternate_ability_enabled"] is True
        assert result["alternate_ability_type"] == "exponential"
        assert result["alternate_ability_args"] == [1.5]
    
    def test_define_alternate_ability_params_logistic(self):
        """Test define_alternate_ability_params with logistic function."""
        config = EvaluationConfig(
            alternate_ability=AlternateAbilityConfig(
                enabled=True,
                function_type="logistic",
                parameters=[5.0, -1.0]  # threshold, slope
            )
        )
        
        result = define_alternate_ability_params(config)
        
        assert result["alternate_ability_enabled"] is True
        assert result["alternate_ability_type"] == "logistic"
        assert result["alternate_ability_args"] == [5.0, -1.0]
    
    def test_define_alternate_ability_params_validation_error(self):
        """Test define_alternate_ability_params with invalid function type."""
        # Validation now happens at the AlternateAbilityConfig level
        with pytest.raises(ValueError):
            AlternateAbilityConfig(
                enabled=True,
                function_type="invalid_function",
                parameters=[1.0]
            )


class TestCalculateEvaluationForecastIntegration:
    """Test integration of bias and ability parameters in calculate_evaluation_forecast."""
    
    def create_test_scenario(self) -> EvaluationScenario:
        """Create a test evaluation scenario."""
        ability = AbilityForecast(
            date=datetime(2025, 1, 1),
            threshold=5.0,
            slope=-1.0,
            scenario="test_scenario",
            model="test_model"
        )
        
        return EvaluationScenario(
            ability=ability,
            doubling_rate=2.0,
            budget_fraction=0.5,
            scenario_id="test_scenario_50pct",
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="static_50pct",
            cost_model="test_cost_model"
        )
    
    def test_calculate_evaluation_forecast_with_elicitation_bias(self):
        """Test calculate_evaluation_forecast includes elicitation bias parameters."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source={"type": "linear", "args": [0.0, 1.0]},
                enabled=True
            )
        )
        
        scenario = self.create_test_scenario()
        design = EvaluationDesign(
            sampler_type=TaskSamplerType.UNIFORM,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            repeats_per_unit=20
        )
        
        forecast = calculate_evaluation_forecast(scenario, design, config)
        
        assert hasattr(forecast, 'elicitation_bias_enabled')
        assert hasattr(forecast, 'elicitation_bias_type')
        assert hasattr(forecast, 'elicitation_bias_args')
        assert forecast.elicitation_bias_enabled is True
        assert forecast.elicitation_bias_type == "linear"
        assert forecast.elicitation_bias_args == [0.0, 1.0]
    
    def test_calculate_evaluation_forecast_with_alternate_ability(self):
        """Test calculate_evaluation_forecast includes alternate ability parameters."""
        config = EvaluationConfig(
            alternate_ability=AlternateAbilityConfig(
                enabled=True,
                function_type="power_law",
                parameters=[2.0]
            )
        )
        
        scenario = self.create_test_scenario()
        design = EvaluationDesign(
            sampler_type=TaskSamplerType.UNIFORM,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            repeats_per_unit=20
        )
        
        forecast = calculate_evaluation_forecast(scenario, design, config)
        
        assert hasattr(forecast, 'alternate_ability_enabled')
        assert hasattr(forecast, 'alternate_ability_type')
        assert hasattr(forecast, 'alternate_ability_args')
        assert forecast.alternate_ability_enabled is True
        assert forecast.alternate_ability_type == "power_law"
        assert forecast.alternate_ability_args == [2.0]
    
    def test_calculate_evaluation_forecast_with_both_features(self):
        """Test calculate_evaluation_forecast with both elicitation bias and alternate ability."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source={"type": "fall_past_threshold", "args": [0.3]},
                enabled=True
            ),
            alternate_ability=AlternateAbilityConfig(
                enabled=True,
                function_type="exponential",
                parameters=[1.2]
            )
        )
        
        scenario = self.create_test_scenario()
        design = EvaluationDesign(
            sampler_type=TaskSamplerType.UNIFORM,
            adjustment_method=WindowAdjustmentMethod.UPPER_BOUND,
            repeats_per_unit=20
        )
        
        forecast = calculate_evaluation_forecast(scenario, design, config)
        
        # Test elicitation bias parameters
        assert forecast.elicitation_bias_enabled is True
        assert forecast.elicitation_bias_type == "fall_past_threshold"
        assert forecast.elicitation_bias_args == [0.3]
        
        # Test alternate ability parameters
        assert forecast.alternate_ability_enabled is True
        assert forecast.alternate_ability_type == "exponential"
        assert forecast.alternate_ability_args == [1.2]


class TestParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_elicitation_bias_invalid_type(self):
        """Test validation of invalid elicitation bias types."""
        with pytest.raises(ValueError):
            ElicitationBiasConfig(
                source={"type": "invalid_type", "args": [0.5]},
                enabled=True
            )
    
    def test_alternate_ability_missing_parameters(self):
        """Test validation when alternate ability is enabled but missing parameters."""
        with pytest.raises(ValueError):
            AlternateAbilityConfig(
                enabled=True,
                function_type="exponential",
                parameters=None  # Should be required when enabled
            )


if __name__ == "__main__":
    pytest.main([__file__])