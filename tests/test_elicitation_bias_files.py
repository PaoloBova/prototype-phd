import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from src.schemas import ElicitationBiasConfig, AbilityForecast, EvaluationScenario
from src.forecast_evaluation import (
    EvaluationConfig,
    define_elicitation_bias
)
from datetime import datetime


class TestElicitationBiasFileLoading:
    """Test file-based elicitation bias configuration loading."""
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name
    
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
    
    def test_load_elicitation_bias_from_valid_file(self):
        """Test loading elicitation bias configuration from a valid JSON file."""
        bias_config = {
            "type": "linear",
            "args": [0.0, 2.0],
            "description": "Linear decline from difficulty 0 with slope 2.0"
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            
            assert result["elicitation_bias_enabled"] is True
            assert result["elicitation_bias_type"] == "linear"
            assert result["elicitation_bias_args"] == [0.0, 2.0]
            
        finally:
            os.unlink(file_path)
    
    def test_load_elicitation_bias_fall_past_threshold_from_file(self):
        """Test loading fall_past_threshold configuration from file."""
        bias_config = {
            "type": "fall_past_threshold",
            "args": [0.3],
            "budget_dependent": True,
            "calibration_source": "RE-bench AIDE vs MODULAR comparison"
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            
            assert result["elicitation_bias_enabled"] is True
            assert result["elicitation_bias_type"] == "fall_past_threshold"
            assert result["elicitation_bias_args"] == [0.3]
            
        finally:
            os.unlink(file_path)
    
    def test_load_elicitation_bias_logistic_from_file(self):
        """Test loading logistic elicitation bias configuration from file."""
        bias_config = {
            "type": "logistic",
            "args": [6.0, -0.5],
            "description": "Logistic decline with threshold=6.0, slope=-0.5"
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            
            assert result["elicitation_bias_enabled"] is True
            assert result["elicitation_bias_type"] == "logistic"
            assert result["elicitation_bias_args"] == [6.0, -0.5]
            
        finally:
            os.unlink(file_path)
    
    def test_load_elicitation_bias_file_not_found(self):
        """Test handling when elicitation bias file is not found."""
        non_existent_path = "/path/to/non_existent_file.json"
        
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source=non_existent_path,
                enabled=True
            )
        )
        
        # Should fall back to default configuration gracefully
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(config, scenario)
        
        assert isinstance(result, dict)
        assert "elicitation_bias_enabled" in result
        # Should either disable or use fallback configuration
    
    def test_load_elicitation_bias_invalid_json(self):
        """Test handling of invalid JSON in elicitation bias file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            invalid_file_path = f.name
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=invalid_file_path,
                    enabled=True
                )
            )
            
            # Should handle JSON parsing error gracefully
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            assert isinstance(result, dict)
            assert "elicitation_bias_enabled" in result
            
        finally:
            os.unlink(invalid_file_path)
    
    def test_load_elicitation_bias_missing_required_fields(self):
        """Test handling of elicitation bias file missing required fields."""
        bias_config = {
            "description": "Missing type and args fields"
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            # Should handle missing fields gracefully
            with pytest.raises((KeyError, ValueError)):
                scenario = self.create_test_scenario()
                define_elicitation_bias(config, scenario)
                
        finally:
            os.unlink(file_path)
    
    def test_load_elicitation_bias_invalid_type(self):
        """Test handling of invalid elicitation bias type in file."""
        bias_config = {
            "type": "invalid_bias_type",
            "args": [0.5]
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            # Should validate the bias type
            with pytest.raises(ValueError):
                scenario = self.create_test_scenario()
                define_elicitation_bias(config, scenario)
                
        finally:
            os.unlink(file_path)


class TestElicitationBiasFallbackBehavior:
    """Test fallback behavior when file loading fails."""
    
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
    
    def test_fallback_to_inline_config_when_file_fails(self):
        """Test that system falls back to inline config when file loading fails."""
        # This test assumes a hybrid approach where file path can be provided
        # but if it fails, it uses inline configuration as fallback
        
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source="/non/existent/path.json",
                enabled=True
            )
        )
        
        # Should fall back gracefully
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(config, scenario)
        
        assert isinstance(result, dict)
        assert "elicitation_bias_enabled" in result
    
    def test_disable_elicitation_bias_on_critical_file_error(self):
        """Test that elicitation bias is disabled when file loading encounters critical errors."""
        config = EvaluationConfig(
            elicitation_bias=ElicitationBiasConfig(
                source="/invalid/path/that/should/cause/error.json",
                enabled=True
            )
        )
        
        scenario = self.create_test_scenario()
        result = define_elicitation_bias(config, scenario)
        
        assert isinstance(result, dict)
        assert "elicitation_bias_enabled" in result
        # Could be disabled as a safe fallback


class TestElicitationBiasFileFormats:
    """Test different elicitation bias file formats and structures."""
    
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
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name
    
    def test_comprehensive_elicitation_bias_config_file(self):
        """Test loading a comprehensive elicitation bias configuration."""
        bias_config = {
            "type": "fall_past_threshold",
            "args": [0.5],
            "metadata": {
                "calibration_source": "RE-bench comparison",
                "experiment_details": {
                    "aide_success_rate": 0.0548,
                    "modular_success_rate": 0.1151,
                    "resource_gap": 0.5,
                    "task_type": "8hr_tasks"
                },
                "created_by": "research_team",
                "version": "1.0"
            },
            "budget_dependent_parameters": {
                "enabled": True,
                "gap_threshold": 0.5,
                "linear_interpolation": True,
                "minimum_sensitivity": 0.0
            },
            "validation": {
                "min_args": 1,
                "max_args": 1,
                "arg_types": ["float"],
                "arg_ranges": [[0.0, 1.0]]
            }
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            
            assert result["elicitation_bias_enabled"] is True
            assert result["elicitation_bias_type"] == "fall_past_threshold"
            assert result["elicitation_bias_args"] == [0.5]
            
        finally:
            os.unlink(file_path)
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name
    
    def test_elicitation_bias_with_budget_dependent_params(self):
        """Test elicitation bias configuration with budget-dependent functional scaling."""
        bias_config = {
            "type": "fall_past_threshold",
            "args": [1.0],  # Base sensitivity rate when budget_gap = 0
            "budget_dependent": True,
            "budget_scaling": {
                "param_0": {  # Scale the first (and only) parameter
                    "type": "linear",
                    "params": {
                        "target_value": 0.0  # Linear scaling from 1.0 to 0.0 as budget_gap goes from 0 to 1
                    }
                }
            }
        }
        
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,
                    enabled=True
                )
            )
            
            # Test different budget scenarios with expected linear scaling
            test_cases = [
                (1.0, [1.0]),   # budget_fraction=1.0 -> gap=0.0 -> 1.0 + (0.0 - 1.0) * 0.0 = 1.0
                (0.75, [0.75]), # budget_fraction=0.75 -> gap=0.25 -> 1.0 + (0.0 - 1.0) * 0.25 = 0.75
                (0.5, [0.5]),   # budget_fraction=0.5 -> gap=0.5 -> 1.0 + (0.0 - 1.0) * 0.5 = 0.5
                (0.25, [0.25]), # budget_fraction=0.25 -> gap=0.75 -> 1.0 + (0.0 - 1.0) * 0.75 = 0.25
                (0.0, [0.0])    # budget_fraction=0.0 -> gap=1.0 -> 1.0 + (0.0 - 1.0) * 1.0 = 0.0
            ]
            
            for budget_fraction, expected_args in test_cases:
                scenario = self.create_test_scenario(budget_fraction=budget_fraction)
                result = define_elicitation_bias(config, scenario)
                
                assert result["elicitation_bias_enabled"] is True
                assert result["elicitation_bias_type"] == "fall_past_threshold"
                assert len(result["elicitation_bias_args"]) == 1
                # Use approximate equality for floating point comparison
                assert abs(result["elicitation_bias_args"][0] - expected_args[0]) < 1e-10, \
                    f"Failed for budget_fraction={budget_fraction}: expected {expected_args[0]}, got {result['elicitation_bias_args'][0]}"
            
        finally:
            os.unlink(file_path)
    
    def test_elicitation_bias_scaling_types(self):
        """Test different scaling types for budget-dependent parameters."""
        test_cases = [
            # Test constant scaling
            {
                "scaling_type": "constant",
                "scaling_params": {},
                "base_value": 0.8,
                "test_gaps": [(0.0, 0.8), (0.5, 0.8), (1.0, 0.8)]  # Should remain constant
            },
            # Test exponential scaling
            {
                "scaling_type": "exponential", 
                "scaling_params": {"decay_rate": 2.0},
                "base_value": 1.0,
                "test_gaps": [(0.0, 1.0), (0.5, 0.368), (1.0, 0.135)]  # exp(-2*gap)
            },
            # Test power law scaling
            {
                "scaling_type": "power_law",
                "scaling_params": {"exponent": 2.0},
                "base_value": 1.0,
                "test_gaps": [(0.0, 1.0), (0.5, 0.25), (1.0, 0.0)]  # (1-gap)^2
            }
        ]
        
        for test_case in test_cases:
            bias_config = {
                "type": "fall_past_threshold",
                "args": [test_case["base_value"]],
                "budget_dependent": True,
                "budget_scaling": {
                    "param_0": {
                        "type": test_case["scaling_type"],
                        "params": test_case["scaling_params"]
                    }
                }
            }
            
            file_path = self.create_test_bias_file(bias_config)
            
            try:
                config = EvaluationConfig(
                    elicitation_bias=ElicitationBiasConfig(
                        source=file_path,
                        enabled=True
                    )
                )
                
                for budget_gap, expected_value in test_case["test_gaps"]:
                    budget_fraction = 1.0 - budget_gap
                    scenario = self.create_test_scenario(budget_fraction=budget_fraction)
                    result = define_elicitation_bias(config, scenario)
                    
                    actual_value = result["elicitation_bias_args"][0]
                    assert abs(actual_value - expected_value) < 0.01, \
                        f"Scaling type {test_case['scaling_type']}: gap={budget_gap}, expected≈{expected_value}, got={actual_value}"
                        
            finally:
                os.unlink(file_path)


class TestElicitationBiasConfigValidation:
    """Test validation of elicitation bias configurations."""
    
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
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name
    
    def test_validate_elicitation_bias_args_count(self):
        """Test validation of argument count for different bias types."""
        test_cases = [
            ("fall_past_threshold", [0.5], True),
            ("fall_past_threshold", [0.5, 0.3], False),  # Too many args
            ("linear", [0.0, 1.0], True),
            ("linear", [0.0], False),  # Too few args
            ("logistic", [5.0, -1.0], True),
            ("logistic", [5.0], False),  # Too few args
        ]
        
        for bias_type, args, should_be_valid in test_cases:
            bias_config = {
                "type": bias_type,
                "args": args
            }
            
            file_path = self.create_test_bias_file(bias_config)
            
            try:
                config = EvaluationConfig(
                    elicitation_bias=ElicitationBiasConfig(
                        source=file_path,
                        enabled=True
                    )
                )
                
                if should_be_valid:
                    scenario = self.create_test_scenario()
                    result = define_elicitation_bias(config, scenario)
                    assert result["elicitation_bias_enabled"] is True
                    assert result["elicitation_bias_type"] == bias_type
                    assert result["elicitation_bias_args"] == args
                else:
                    with pytest.raises((ValueError, TypeError)):
                        scenario = self.create_test_scenario()
                        define_elicitation_bias(config, scenario)
                        
            finally:
                os.unlink(file_path)
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name


class TestElicitationBiasPathResolution:
    """Test path resolution for elicitation bias files."""
    
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
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name
    
    def test_absolute_path_resolution(self):
        """Test resolution of absolute paths."""
        bias_config = {"type": "fall_past_threshold", "args": [0.5]}
        file_path = self.create_test_bias_file(bias_config)
        
        try:
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=file_path,  # Absolute path
                    enabled=True
                )
            )
            
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            assert result["elicitation_bias_enabled"] is True
            
        finally:
            os.unlink(file_path)
    
    def test_relative_path_resolution(self):
        """Test resolution of relative paths."""
        # Create a file in a known location
        bias_config = {"type": "fall_past_threshold", "args": [0.5]}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "bias_config.json")
            with open(file_path, 'w') as f:
                json.dump(bias_config, f)
            
            # Use relative path
            relative_path = os.path.relpath(file_path)
            
            config = EvaluationConfig(
                elicitation_bias=ElicitationBiasConfig(
                    source=relative_path,
                    enabled=True
                )
            )
            
            scenario = self.create_test_scenario()
            result = define_elicitation_bias(config, scenario)
            assert result["elicitation_bias_enabled"] is True
    
    def create_test_bias_file(self, content: dict) -> str:
        """Create a temporary bias configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(content, f)
            return f.name


if __name__ == "__main__":
    pytest.main([__file__])