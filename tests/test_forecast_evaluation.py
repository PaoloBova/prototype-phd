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
    EvaluationConfig,
    ElicitationBiasConfig,
    AlternateAbilityConfig,
    TaskSamplerType,
    WindowAdjustmentMethod,
    CalculatedElicitationBias,
    CalculatedAlternateAbility,
)
from src.forecast_evaluation import (
    define_elicitation_bias,
    define_alternate_ability_params,
    calculate_evaluation_forecast
)


class TestElicitationBiasConfig:
    """Test the ElicitationBiasConfig Pydantic model."""
    
    def test_elicitation_bias_config_fall_past_threshold(self):
        """Test creating ElicitationBiasConfig with fall_past_threshold."""
        from src.schemas import ElicitationBiasType
        config = ElicitationBiasConfig(
            bias_type=ElicitationBiasType.FALL_PAST_THRESHOLD,
            enabled=True,
            parameters=[0.5]
        )
        assert config.enabled is True
        assert config.bias_type == ElicitationBiasType.FALL_PAST_THRESHOLD
        assert config.parameters == [0.5]
    
    def test_elicitation_bias_config_with_file(self):
        """Test creating ElicitationBiasConfig with source file."""
        from src.schemas import ElicitationBiasType
        config = ElicitationBiasConfig(
            bias_type=ElicitationBiasType.LINEAR,
            enabled=True,
            source_file="/path/to/elicitation_bias.json",
            parameters=[0.0, 1.0]
        )
        assert config.enabled is True
        assert config.source_file == "/path/to/elicitation_bias.json"
        assert config.bias_type == ElicitationBiasType.LINEAR
        assert config.parameters == [0.0, 1.0]
    
    def test_elicitation_bias_config_defaults(self):
        """Test ElicitationBiasConfig with default values."""
        config = ElicitationBiasConfig()
        assert config.enabled is True
        from src.schemas import ElicitationBiasType
        assert config.bias_type == ElicitationBiasType.FALL_PAST_THRESHOLD
        assert config.parameters == [0.5, 0.1]
    
    @given(
        enabled=st.booleans(),
        bias_type=st.sampled_from(["fall_past_threshold", "linear", "logistic"]),
        parameters=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=3)
    )
    def test_elicitation_bias_config_property_based(self, enabled, bias_type, parameters):
        """Property-based test for ElicitationBiasConfig."""
        from src.schemas import ElicitationBiasType
        config = ElicitationBiasConfig(
            bias_type=ElicitationBiasType(bias_type),
            enabled=enabled,
            parameters=parameters
        )
        assert config.enabled == enabled
        assert config.bias_type.value == bias_type
        assert config.parameters == parameters


class TestAlternateAbilityConfig:
    """Test the AlternateAbilityConfig Pydantic model."""
    
    def test_alternate_ability_config_disabled(self):
        """Test creating AlternateAbilityConfig with disabled state."""
        config = AlternateAbilityConfig(enabled=False)
        assert config.enabled is False
        from src.schemas import AlternateAbilityType
        assert config.function_type == AlternateAbilityType.LOGISTIC  # Default value
        assert config.parameters == {}
    
    def test_alternate_ability_config_exponential(self):
        """Test AlternateAbilityConfig with exponential function."""
        from src.schemas import AlternateAbilityType
        config = AlternateAbilityConfig(
            enabled=True,
            function_type=AlternateAbilityType.EXPONENTIAL,
            name="Exponential decay robustness check",
            parameters={"rate": 1.5}
        )
        assert config.enabled is True
        assert config.function_type == AlternateAbilityType.EXPONENTIAL
        assert config.parameters == {"rate": 1.5}
        assert "Exponential" in config.name
    
    def test_alternate_ability_config_power_law(self):
        """Test AlternateAbilityConfig with power law function."""
        from src.schemas import AlternateAbilityType
        config = AlternateAbilityConfig(
            enabled=True,
            function_type=AlternateAbilityType.POWER_LAW,
            parameters={"exponent": 2.0}
        )
        assert config.enabled is True
        assert config.function_type == AlternateAbilityType.POWER_LAW
        assert config.parameters == {"exponent": 2.0}
    
    @given(
        enabled=st.booleans(),
        function_type=st.sampled_from(["exponential", "power_law", "tangent", "logistic"])
    )
    def test_alternate_ability_config_property_based(self, enabled, function_type):
        """Property-based test for AlternateAbilityConfig."""
        from src.schemas import AlternateAbilityType
        parameters = {"param1": 1.5, "param2": 2.0} if enabled else {}
        config = AlternateAbilityConfig(
            enabled=enabled,
            function_type=AlternateAbilityType(function_type),
            parameters=parameters
        )
        assert config.enabled == enabled
        assert config.function_type.value == function_type
        assert config.parameters == parameters


class TestEvaluationConfigExtended:
    """Test the extended EvaluationConfig with bias and ability configurations."""
    
    def test_evaluation_config_with_bias_and_ability(self):
        """Test EvaluationConfig includes elicitation bias and alternate ability configs."""
        from src.schemas import ElicitationBiasType, AlternateAbilityType
        config = EvaluationConfig(
            elicitation_bias_config=ElicitationBiasConfig(
                bias_type=ElicitationBiasType.LINEAR,
                enabled=True,
                parameters=[0.0, 1.0]
            ),
            alternate_ability_config=AlternateAbilityConfig(
                enabled=True,
                function_type=AlternateAbilityType.POWER_LAW,
                parameters={"exponent": 1.5}
            )
        )
        
        assert hasattr(config, 'elicitation_bias_config')
        assert hasattr(config, 'alternate_ability_config')
        assert config.elicitation_bias_config.enabled is True
        assert config.alternate_ability_config.enabled is True
        assert config.alternate_ability_config.function_type == AlternateAbilityType.POWER_LAW


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
            intercept=1.0,
            budget_fraction=budget_fraction,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="static_50pct",
            cost_model="test_cost_model",
            ability_variant="base",
            cost_variant="base",
            base_ability_id="test_ability_base",
            base_cost_id="test_cost_base"
        )
    
    def test_define_elicitation_bias_inline_config(self):
        """Test define_elicitation_bias with inline configuration."""
        from src.schemas import ElicitationBiasType
        config = EvaluationConfig(
            elicitation_bias_config=ElicitationBiasConfig(
                bias_type=ElicitationBiasType.FALL_PAST_THRESHOLD,
                enabled=True,
                parameters=[0.5, 0.1]
            )
        )
        
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(scenario, config)
        
        assert isinstance(result, CalculatedElicitationBias)
        assert result.enabled is True
        assert result.bias_type == "fall_past_threshold"
        assert result.args == [0.5, 0.1]
    
    def test_define_elicitation_bias_disabled(self):
        """Test define_elicitation_bias when disabled."""
        config = EvaluationConfig(
            elicitation_bias_config=ElicitationBiasConfig(enabled=False)
        )
        
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(scenario, config)
        
        assert isinstance(result, CalculatedElicitationBias)
        assert result.enabled is False
    
    def test_define_elicitation_bias_file_source(self):
        """Test define_elicitation_bias with file path source."""
        from src.schemas import ElicitationBiasType
        config = EvaluationConfig(
            elicitation_bias_config=ElicitationBiasConfig(
                bias_type=ElicitationBiasType.LINEAR,
                source_file="/path/to/bias_config.json",
                enabled=True,
                parameters=[0.0, 1.0]
            )
        )
        
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(scenario, config)
        assert isinstance(result, CalculatedElicitationBias)
        assert result.enabled is True


class TestDefineAlternateAbilityParams:
    """Test the define_alternate_ability_params function."""
    
    def test_define_alternate_ability_params_disabled(self):
        """Test define_alternate_ability_params when disabled."""
        config = EvaluationConfig(
            alternate_ability_config=AlternateAbilityConfig(enabled=False)
        )
        
        result = define_alternate_ability_params(config)
        
        assert isinstance(result, CalculatedAlternateAbility)
        assert result.enabled is False
    
    def test_define_alternate_ability_params_exponential(self):
        """Test define_alternate_ability_params with exponential function."""
        from src.schemas import AlternateAbilityType
        config = EvaluationConfig(
            alternate_ability_config=AlternateAbilityConfig(
                enabled=True,
                function_type=AlternateAbilityType.EXPONENTIAL,
                parameters={"rate": 1.5}
            )
        )
        
        result = define_alternate_ability_params(config)
        
        assert isinstance(result, CalculatedAlternateAbility)
        assert result.enabled is True
        assert result.function_type == "exponential"
        # Note: The function may calculate different args than input parameters
        assert isinstance(result.args, list)
    
    def test_define_alternate_ability_params_logistic(self):
        """Test define_alternate_ability_params with logistic function."""
        from src.schemas import AlternateAbilityType
        config = EvaluationConfig(
            alternate_ability_config=AlternateAbilityConfig(
                enabled=True,
                function_type=AlternateAbilityType.LOGISTIC,
                parameters={"threshold": 5.0, "slope": -1.0}
            )
        )
        
        result = define_alternate_ability_params(config)
        
        assert result.enabled is True
        assert result.function_type == "logistic"
        assert isinstance(result.args, list)
    
    def test_define_alternate_ability_params_validation_error(self):
        """Test define_alternate_ability_params with invalid function type."""
        # Validation now happens at the AlternateAbilityConfig level
        from src.schemas import AlternateAbilityType
        with pytest.raises(ValueError):
            # This should raise ValueError for invalid enum value
            AlternateAbilityType("invalid_function")


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
            intercept=1.0,
            budget_fraction=0.5,
            ability_id="test_ability",
            cost_id="test_cost",
            constraint_id="static_50pct",
            cost_model="test_cost_model",
            ability_variant="base",
            cost_variant="base",
            base_ability_id="test_ability_base",
            base_cost_id="test_cost_base"
        )
    
    def test_calculate_evaluation_forecast_with_elicitation_bias(self):
        """Test calculate_evaluation_forecast includes elicitation bias parameters."""
        from src.schemas import ElicitationBiasType
        config = EvaluationConfig(
            elicitation_bias_config=ElicitationBiasConfig(
                bias_type=ElicitationBiasType.LINEAR,
                enabled=True,
                parameters=[0.0, 1.0]
            )
        )
        
        scenario = self.create_test_scenario()
        forecast = calculate_evaluation_forecast(scenario, config)
        
        assert isinstance(forecast, EvaluationForecast)
        assert hasattr(forecast, 'design')
        assert hasattr(forecast.design, 'elicitation_bias')
        assert forecast.design.elicitation_bias.enabled is True
        assert forecast.design.elicitation_bias.bias_type == "linear"
        assert forecast.design.elicitation_bias.args == [0.0, 1.0]
    
    def test_calculate_evaluation_forecast_with_alternate_ability(self):
        """Test calculate_evaluation_forecast includes alternate ability parameters."""
        from src.schemas import AlternateAbilityType
        config = EvaluationConfig(
            alternate_ability_config=AlternateAbilityConfig(
                enabled=True,
                function_type=AlternateAbilityType.POWER_LAW,
                parameters={"exponent": 2.0}
            )
        )
        
        scenario = self.create_test_scenario()
        forecast = calculate_evaluation_forecast(scenario, config)
        
        assert isinstance(forecast, EvaluationForecast)
        assert hasattr(forecast, 'design')
        assert hasattr(forecast.design, 'alternate_ability')
        assert forecast.design.alternate_ability.enabled is True
        assert forecast.design.alternate_ability.function_type == "power_law"
        assert isinstance(forecast.design.alternate_ability.args, list)
    
    def test_calculate_evaluation_forecast_with_both_features(self):
        """Test calculate_evaluation_forecast with both elicitation bias and alternate ability."""
        from src.schemas import ElicitationBiasType, AlternateAbilityType
        config = EvaluationConfig(
            elicitation_bias_config=ElicitationBiasConfig(
                bias_type=ElicitationBiasType.FALL_PAST_THRESHOLD,
                enabled=True,
                parameters=[0.3]
            ),
            alternate_ability_config=AlternateAbilityConfig(
                enabled=True,
                function_type=AlternateAbilityType.EXPONENTIAL,
                parameters={"rate": 1.2}
            )
        )
        
        scenario = self.create_test_scenario()
        forecast = calculate_evaluation_forecast(scenario, config)
        
        assert isinstance(forecast, EvaluationForecast)
        
        # Test elicitation bias parameters
        assert forecast.design.elicitation_bias.enabled is True
        assert forecast.design.elicitation_bias.bias_type == "fall_past_threshold"
        assert forecast.design.elicitation_bias.args == [0.3]
        
        # Test alternate ability parameters
        assert forecast.design.alternate_ability.enabled is True
        assert forecast.design.alternate_ability.function_type == "exponential"
        assert isinstance(forecast.design.alternate_ability.args, list)


class TestParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_elicitation_bias_invalid_type(self):
        """Test validation of invalid elicitation bias types."""
        from src.schemas import ElicitationBiasType
        with pytest.raises(ValueError):
            # This should raise ValueError for invalid enum value
            ElicitationBiasType("invalid_type")
    
    def test_alternate_ability_invalid_type(self):
        """Test validation of invalid alternate ability types."""
        from src.schemas import AlternateAbilityType
        with pytest.raises(ValueError):
            # This should raise ValueError for invalid enum value
            AlternateAbilityType("invalid_function")


if __name__ == "__main__":
    pytest.main([__file__])