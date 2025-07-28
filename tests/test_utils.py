import unittest
from prototype_phd.utils import expand_sweep_config, _set_nested_value_by_path


class TestExpandSweepConfig(unittest.TestCase):
    """Test suite for the expand_sweep_config function."""
    
    def setUp(self):
        """Set up test data for each test."""
        self.simple_config = {
            "base_config": {
                "model": {"lr": 0.01, "layers": 64},
                "training": {"epochs": 100}
            },
            "_sweep": [
                {"path": ["model", "lr"], "values": [0.01, 0.001]},
                {"path": ["training", "epochs"], "values": [100, 200]}
            ]
        }
        
        self.nested_config = {
            "base_config": {
                "model": {
                    "params": {"lr": 0.01, "dropout": 0.1},
                    "layers": [64, 32, 16]
                },
                "data": {"batch_size": 32}
            },
            "_sweep": [
                {"path": ["model", "params", "lr"], "values": [0.01, 0.001, 0.0001]},
                {"path": ["model", "layers", 0], "values": [64, 128]},
                {"path": ["data", "batch_size"], "values": [16, 32, 64]}
            ]
        }
    
    def test_basic_functionality(self):
        """Test basic parameter sweep functionality."""
        configs = expand_sweep_config(self.simple_config)
        
        # Should generate 2 * 2 = 4 configurations
        self.assertEqual(len(configs), 4)
        
        # Check that all configs have the expected structure
        for config in configs:
            self.assertIn("model", config)
            self.assertIn("training", config)
            self.assertIn("_sweep_metadata", config)
            self.assertIn("parameter_combination", config["_sweep_metadata"])
            self.assertIn("combination_index", config["_sweep_metadata"])
        
        # Test specific combinations
        lr_values = [c["model"]["lr"] for c in configs]
        epoch_values = [c["training"]["epochs"] for c in configs]
        
        self.assertIn(0.01, lr_values)
        self.assertIn(0.001, lr_values)
        self.assertIn(100, epoch_values)
        self.assertIn(200, epoch_values)
    
    def test_nested_paths_with_array_indices(self):
        """Test parameter sweep with nested paths and array indices."""
        configs = expand_sweep_config(self.nested_config)
        
        # Should generate 3 * 2 * 3 = 18 configurations
        self.assertEqual(len(configs), 18)
        
        # Check that array indices work correctly
        layer_0_values = [c["model"]["layers"][0] for c in configs]
        self.assertIn(64, layer_0_values)
        self.assertIn(128, layer_0_values)
        
        # Check that nested dict paths work correctly
        lr_values = [c["model"]["params"]["lr"] for c in configs]
        self.assertIn(0.01, lr_values)
        self.assertIn(0.001, lr_values)
        self.assertIn(0.0001, lr_values)
    
    def test_metadata_generation(self):
        """Test that metadata is correctly generated for each configuration."""
        configs = expand_sweep_config(self.simple_config)
        
        for i, config in enumerate(configs):
            metadata = config["_sweep_metadata"]
            
            # Check combination index
            self.assertEqual(metadata["combination_index"], i)
            
            # Check parameter combination
            param_combo = metadata["parameter_combination"]
            self.assertIn("model.lr", param_combo)
            self.assertIn("training.epochs", param_combo)
            
            # Verify the parameter values match the config
            self.assertEqual(param_combo["model.lr"], config["model"]["lr"])
            self.assertEqual(param_combo["training.epochs"], config["training"]["epochs"])
    
    def test_deep_copy_isolation(self):
        """Test that configurations are properly isolated from each other."""
        configs = expand_sweep_config(self.simple_config)
        
        # Modify one config
        configs[0]["model"]["lr"] = 999
        configs[0]["model"]["new_param"] = "test"
        
        # Check that other configs are not affected
        for i in range(1, len(configs)):
            self.assertNotEqual(configs[i]["model"]["lr"], 999)
            self.assertNotIn("new_param", configs[i]["model"])
        
        # Check that original base_config is not modified
        original_lr = self.simple_config["base_config"]["model"]["lr"]
        self.assertEqual(original_lr, 0.01)
    
    def test_single_parameter_sweep(self):
        """Test sweep with only one parameter."""
        single_param_config = {
            "base_config": {"param": 1},
            "_sweep": [{"path": ["param"], "values": [1, 2, 3, 4, 5]}]
        }
        
        configs = expand_sweep_config(single_param_config)
        self.assertEqual(len(configs), 5)
        
        param_values = [c["param"] for c in configs]
        self.assertEqual(sorted(param_values), [1, 2, 3, 4, 5])
    
    def test_empty_sweep(self):
        """Test behavior with empty sweep specification."""
        empty_sweep_config = {
            "base_config": {"param": 1},
            "_sweep": []
        }
        
        configs = expand_sweep_config(empty_sweep_config)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["param"], 1)
        self.assertIn("_sweep_metadata", configs[0])
    
    def test_error_handling_missing_base_config(self):
        """Test error handling when base_config is missing."""
        invalid_config = {"_sweep": []}
        
        with self.assertRaises(ValueError) as context:
            expand_sweep_config(invalid_config)
        
        self.assertIn("base_config", str(context.exception))
    
    def test_error_handling_missing_sweep(self):
        """Test error handling when _sweep is missing."""
        invalid_config = {"base_config": {}}
        
        with self.assertRaises(ValueError) as context:
            expand_sweep_config(invalid_config)
        
        self.assertIn("_sweep", str(context.exception))
    
    def test_error_handling_invalid_sweep_format(self):
        """Test error handling for invalid sweep specifications."""
        # Test non-list _sweep
        invalid_config1 = {
            "base_config": {},
            "_sweep": {"not": "a list"}
        }
        
        with self.assertRaises(TypeError) as context:
            expand_sweep_config(invalid_config1)
        
        self.assertIn("must be a list", str(context.exception))
        
        # Test missing path in sweep spec
        invalid_config2 = {
            "base_config": {},
            "_sweep": [{"values": [1, 2]}]
        }
        
        with self.assertRaises(ValueError) as context:
            expand_sweep_config(invalid_config2)
        
        self.assertIn("missing 'path' key", str(context.exception))
        
        # Test missing values in sweep spec
        invalid_config3 = {
            "base_config": {},
            "_sweep": [{"path": ["param"]}]
        }
        
        with self.assertRaises(ValueError) as context:
            expand_sweep_config(invalid_config3)
        
        self.assertIn("missing 'values' key", str(context.exception))
    
    def test_error_handling_invalid_path_types(self):
        """Test error handling for invalid path element types."""
        invalid_config = {
            "base_config": {"param": 1},
            "_sweep": [{"path": ["param", 3.14], "values": [1, 2]}]
        }
        
        with self.assertRaises(TypeError) as context:
            expand_sweep_config(invalid_config)
        
        self.assertIn("must be string or integer", str(context.exception))
    
    def test_complex_data_types_in_values(self):
        """Test that complex data types can be used as sweep values."""
        complex_config = {
            "base_config": {
                "optimizer": {"type": "adam"},
                "layers": [64],
                "config": {"nested": {"value": 1}}
            },
            "_sweep": [
                {"path": ["optimizer"], "values": [
                    {"type": "adam", "lr": 0.01},
                    {"type": "sgd", "lr": 0.1}
                ]},
                {"path": ["layers"], "values": [
                    [64, 32],
                    [128, 64, 32]
                ]},
                {"path": ["config"], "values": [
                    {"nested": {"value": 1}},
                    {"nested": {"value": 2, "extra": True}}
                ]}
            ]
        }
        
        configs = expand_sweep_config(complex_config)
        self.assertEqual(len(configs), 2 * 2 * 2)  # 8 configurations
        
        # Check that complex values are properly assigned
        optimizer_types = [c["optimizer"]["type"] for c in configs]
        self.assertIn("adam", optimizer_types)
        self.assertIn("sgd", optimizer_types)
        
        layer_configs = [len(c["layers"]) for c in configs]
        self.assertIn(2, layer_configs)  # [64, 32]
        self.assertIn(3, layer_configs)  # [128, 64, 32]


class TestSetNestedValueByPath(unittest.TestCase):
    """Test suite for the _set_nested_value_by_path helper function."""
    
    def test_simple_dict_path(self):
        """Test setting values in simple dictionary paths."""
        config = {"a": {"b": {"c": 1}}}
        _set_nested_value_by_path(config, ["a", "b", "c"], 999)
        self.assertEqual(config["a"]["b"]["c"], 999)
    
    def test_array_index_path(self):
        """Test setting values using array indices."""
        config = {"array": [1, 2, 3]}
        _set_nested_value_by_path(config, ["array", 1], 999)
        self.assertEqual(config["array"], [1, 999, 3])
    
    def test_mixed_path_types(self):
        """Test paths that mix dictionary keys and array indices."""
        config = {"models": [{"name": "model1"}, {"name": "model2"}]}
        _set_nested_value_by_path(config, ["models", 0, "name"], "new_model")
        self.assertEqual(config["models"][0]["name"], "new_model")
        self.assertEqual(config["models"][1]["name"], "model2")  # Unchanged
    
    def test_single_element_path(self):
        """Test paths with only one element."""
        config = {"param": 1}
        _set_nested_value_by_path(config, ["param"], 999)
        self.assertEqual(config["param"], 999)
    
    def test_negative_array_indices(self):
        """Test that negative array indices work correctly."""
        config = {"array": [1, 2, 3]}
        _set_nested_value_by_path(config, ["array", -1], 999)
        self.assertEqual(config["array"], [1, 2, 999])
    
    def test_error_handling_missing_key(self):
        """Test error handling when dictionary key is missing."""
        config = {"a": {}}
        
        with self.assertRaises(KeyError) as context:
            _set_nested_value_by_path(config, ["a", "missing", "key"], 999)
        
        self.assertIn("missing", str(context.exception))
    
    def test_error_handling_index_out_of_bounds(self):
        """Test error handling when array index is out of bounds."""
        config = {"array": [1, 2, 3]}
        
        with self.assertRaises(IndexError) as context:
            _set_nested_value_by_path(config, ["array", 10], 999)
        
        self.assertIn("out of bounds", str(context.exception))
    
    def test_error_handling_wrong_type_dict(self):
        """Test error handling when expecting dict but finding other type."""
        config = {"param": [1, 2, 3]}
        
        with self.assertRaises(TypeError) as context:
            _set_nested_value_by_path(config, ["param", "key"], 999)
        
        self.assertIn("Expected dict", str(context.exception))
    
    def test_error_handling_wrong_type_list(self):
        """Test error handling when expecting list but finding other type."""
        config = {"param": {"key": "value"}}
        
        with self.assertRaises(TypeError) as context:
            _set_nested_value_by_path(config, ["param", 0], 999)
        
        self.assertIn("Expected list", str(context.exception))


if __name__ == '__main__':
    unittest.main()