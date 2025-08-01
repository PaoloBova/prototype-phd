"""
Pydantic models for the forecast detection case study.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from datetime import datetime
import hashlib
import json


def generate_content_hash(data: Dict[str, Any], prefix: str = "") -> str:
    """
    Generate a reproducible hash ID from the content of a dictionary.
    
    Args:
        data: Dictionary to hash
        prefix: Optional prefix for the hash ID
        
    Returns:
        Hash string with optional prefix
    """
    # Convert to JSON with sorted keys for reproducibility
    json_str = json.dumps(data, sort_keys=True, default=str)
    hash_obj = hashlib.sha256(json_str.encode())
    hash_id = hash_obj.hexdigest()[:12]  # Use first 12 characters
    return f"{prefix}_{hash_id}" if prefix else hash_id


class RawDataRecord(BaseModel):
    """Raw record from detection dataset."""
    task_id: str = Field(..., description="Unique identifier for the task")
    model: str = Field(..., description="Model used for generation")
    alias: Optional[str] = Field(None, description="Alternative name for model")
    task_source: str = Field(..., description="Source of the task")
    human_minutes: float = Field(..., description="Human time to complete task in minutes")
    generation_cost: float = Field(..., description="Cost of generation")
    score_binarized: int = Field(..., description="Binary success indicator (1=success, 0=failure)")

class ProcessedDataRecord(RawDataRecord):
    """Processed record with derived features."""
    human_seconds: float = Field(..., description="Human time in seconds")
    log2_human_seconds: float = Field(..., description="Log2 of human time in seconds")
    bin_power: int = Field(..., description="Discretized bin based on log2 scale")
    date: Optional[datetime] = Field(None, description="Release date of the model")

class LogisticFitParams(BaseModel):
    """Parameters for a logistic curve fit."""
    threshold: float = Field(..., description="Threshold parameter (inflection point)")
    slope: float = Field(..., description="Slope parameter (steepness)")
    model: str = Field(..., description="Model identifier")
    date: datetime = Field(..., description="Date of the fitted data")

class AbilityForecast(BaseModel):
    """Forecasted ability curve parameters."""
    date: datetime = Field(..., description="Forecast date")
    threshold: float = Field(..., description="50% reliability threshold")
    slope: float = Field(..., description="Curve slope")
    scenario: str = Field(..., description="Name of forecast scenario")
    model: str = Field(..., description="Model identifier (for future models, this is a placeholder)")
    threshold_ci_lower: Optional[float] = Field(None, description="Lower bound of 95% confidence interval for threshold")
    threshold_ci_upper: Optional[float] = Field(None, description="Upper bound of 95% confidence interval for threshold")
    slope_ci_lower: Optional[float] = Field(None, description="Lower bound of 95% confidence interval for slope")
    slope_ci_upper: Optional[float] = Field(None, description="Upper bound of 95% confidence interval for slope")


class CostTrend(BaseModel):
    """Cost doubling trend."""
    doubling_rate: float = Field(..., description="Task difficulty units per doubling of cost")
    intercept: float = Field(..., description="Base cost at difficulty 0")
    r_squared: float = Field(..., description="Goodness of fit")

class ResourceConstraintType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"

class ResourceConstraint(BaseModel):
    """Resource constraint configuration."""
    type: ResourceConstraintType = Field(..., description="Type of resource constraint")
    name: str = Field(..., description="Name of the scenario")
    values: Union[float, List[Tuple[datetime, float]]] = Field(
        ..., description="Either fixed value or list of (date, value) tuples for dynamic scenarios"
    )

class TaskSamplerType(str, Enum):
    """Types of task sampling distributions."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    CUSTOM = "custom"

class WindowAdjustmentMethod(str, Enum):
    """Methods for adjusting evaluation windows based on budget constraints."""
    UPPER_BOUND = "upper_bound"  # Adjust upper bound only (default method)
    BOTH_BOUNDS = "both_bounds"  # Adjust both upper and lower bounds
    SAMPLE_BASED = "sample_based"  # Use sampling frequency instead of window adjustment

class EvaluationScenario(BaseModel):
    """A scenario for evaluation combining ability, cost, and resource constraints."""
    ability: AbilityForecast = Field(..., description="Ability forecast for this scenario")
    doubling_rate: float = Field(..., description="Cost doubling rate in difficulty units")
    intercept: float = Field(..., description="Base cost at difficulty 0")
    budget_fraction: float = Field(..., description="Budget as fraction of gold standard")
    ability_id: str = Field(..., description="ID of the ability forecast used")
    cost_id: str = Field(..., description="ID of the cost trend used")
    constraint_id: str = Field(..., description="ID of the resource constraint applied")
    cost_model: str = Field(..., description="Name of the cost model used")
    
    # Variant metadata
    ability_variant: str = Field(..., description="Variant type of ability: base, lower, upper")
    cost_variant: str = Field(..., description="Variant type of cost: base, lower, upper")
    base_ability_id: str = Field(..., description="Base ability ID without variant suffix")
    base_cost_id: str = Field(..., description="Base cost ID without variant suffix")

class ElicitationBiasType(str, Enum):
    """Types of elicitation bias functions."""
    FALL_PAST_THRESHOLD = "fall_past_threshold"
    LINEAR = "linear"
    LOGISTIC = "logistic"
    LOGISTIC_ABILITY_SHIFT = "logistic_ability_shift"
    TASK_FILTER = "task_filter"

class ElicitationBiasConfig(BaseModel):
    """Configuration for elicitation bias parameters."""
    bias_type: ElicitationBiasType = Field(ElicitationBiasType.FALL_PAST_THRESHOLD, description="Type of elicitation bias function to use")
    enabled: bool = Field(False, description="Whether elicitation bias is enabled")
    name: Optional[str] = Field(None, description="Optional name for this configuration")
    source_file: Optional[str] = Field(None, description="Path to data, used by some types of elicitation bias")
    parameters: Dict[str, float] = Field(
        {"sensitivity_rate": 0.5, "threshold": 0.0, "slope": 1.0, "delta": 1.0, "elicitation_threshold": 0.0, "sensitivity_rate_after": 0.5},
        description="Named parameters for elicitation bias functions. Keys: sensitivity_rate (fall_past_threshold), threshold/slope (linear/logistic), delta (logistic_ability_shift), elicitation_threshold/sensitivity_rate_after (task_filter)"
    )
    budget_dependent: bool = Field(False, description="Whether parameters should be scaled based on budget")
    budget_scaling: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Budget scaling configuration for each parameter"
    )

class AlternateAbilityType(str, Enum):
    """Types of alternate ability functions."""
    RICHARDS = "richards_generalized_logistic"
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    LOGISTIC = "logistic"

class AlternateAbility(BaseModel):
    """Configuration for alternate ability function parameters."""
    enabled: bool = Field(False, description="Whether alternate ability function is enabled")
    function_type: AlternateAbilityType = Field(AlternateAbilityType.LOGISTIC, description="Type of alternate ability function")
    name: Optional[str] = Field(None, description="Optional name for this configuration")
    args: Dict[str, Union[float, bool]] = Field({}, description="Parameters for the alternate ability function")

class CalculatedElicitationBias(BaseModel):
    """Calculated elicitation bias with derived parameters."""
    enabled: bool = Field(False, description="Whether elicitation bias is enabled")
    bias_type: ElicitationBiasType = Field(ElicitationBiasType.FALL_PAST_THRESHOLD, description="Type of elicitation bias function to use")
    name: Optional[str] = Field(None, description="Optional name for this configuration")
    source_file: Optional[str] = Field(None, description="Path to data file if used")
    parameters: Dict[str, float] = Field({}, description="Original configuration parameters")
    args: List[float] = Field([], description="Calculated/scaled bias function arguments")

class EvaluationDesign(BaseModel):
    """Evaluation design containing both parameters and calculated results."""
    # Design parameters
    sampler_type: TaskSamplerType = Field(
        TaskSamplerType.UNIFORM, 
        description="Type of task distribution"
    )
    adjustment_method: WindowAdjustmentMethod = Field(
        WindowAdjustmentMethod.UPPER_BOUND, 
        description="Method used to adjust window based on budget"
    )
    repeats_per_unit: int = Field(
        20, 
        description="Number of sample repeats per difficulty unit"
    )
    sampler_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional parameters for the sampler"
    )
    
    # Calculated results 
    window_lower: float = Field(..., description="Lower bound of evaluation window")
    window_upper: float = Field(..., description="Upper bound of evaluation window")
    total_samples: int = Field(..., description="Total number of tasks to sample")
    gold_standard_cost: float = Field(..., description="Total cost of gold standard evaluation")
    available_budget: float = Field(..., description="Available budget (gold_standard_cost * budget_fraction)")
    original_window_lower: float = Field(..., description="Original lower bound of evaluation window")
    original_window_upper: float = Field(..., description="Original upper bound of evaluation window")

    elicitation_bias: CalculatedElicitationBias = Field(..., description="Calculated elicitation bias configuration")
    alternate_ability: AlternateAbility = Field(..., description="Alternate ability")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation forecasts."""
    resource_constraints: Dict[str, Any] = Field(
        {
            "static_budgets": [1.0, 0.75, 0.5, 0.25, 0.1, 0.0],
            "include_dynamic_scenarios": False,
            "dynamic_start_date": "2025-01-01T00:00:00",
            "dynamic_end_date": "2030-12-31T00:00:00",
            "dynamic_frequency": "YE"
        },
        description="Resource constraint parameters"
    )
    scenario_generation: Dict[str, bool] = Field(
        {
            "include_base_scenarios": True,
            "include_ci_scenarios": True
        },
        description="Scenario generation options"
    )

    sampler_type: TaskSamplerType = Field(
        TaskSamplerType.UNIFORM, 
        description="Type of task distribution"
    )
    adjustment_method: WindowAdjustmentMethod = Field(
        WindowAdjustmentMethod.UPPER_BOUND, 
        description="Method used to adjust window based on budget"
    )
    repeats_per_unit: int = Field(
        20, 
        description="Number of sample repeats per difficulty unit"
    )
    sampler_params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional parameters for the sampler"
    )
    
    coverage_ratio: float = Field(
        0.8, 
        description="Coverage ratio for evaluation window (default 80%)"
    )
    
    elicitation_bias_config: ElicitationBiasConfig = Field(
        default_factory=ElicitationBiasConfig,
        description="Elicitation bias configuration for this run"
    )
    alternate_ability: AlternateAbility = Field(
        default_factory=AlternateAbility,
        description="Alternate ability for this run" 
    )

class EvaluationForecast(BaseModel):
    """Clean composition of scenario and design for an evaluation forecast."""
    scenario_id: str = Field(..., description="Hash-based ID of the scenario")
    design_id: str = Field(..., description="Hash-based ID of the design")
    scenario: EvaluationScenario = Field(..., description="Evaluation scenario")
    design: EvaluationDesign = Field(..., description="Evaluation design with calculated results")
    config: EvaluationConfig = Field(..., description="Full evaluation configuration including sweep parameters")

class BootstrapConfig(BaseModel):
    """Configuration for bootstrap analysis."""
    n_bootstrap: int = Field(1000, description="Number of bootstrap samples")
    sample_size: Optional[int] = Field(None, description="Size of each bootstrap sample")
    weights: Optional[List[float]] = Field(None, description="Sampling weights")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")

class SensitivityResult(BaseModel):
    """Results of sensitivity analysis for a single scenario."""
    ability_scenario: str = Field(..., description="Ability forecast scenario")
    budget_scenario: str = Field(..., description="Budget scenario name")
    date: datetime = Field(..., description="Forecast date")
    estimator: str = Field(..., description="Type of estimator used")
    mean: float = Field(..., description="Mean estimate of the parameter")
    true_value: float = Field(..., description="True value of the parameter being estimated")
    bias: float = Field(..., description="Mean difference between estimate and true value")
    variance: float = Field(..., description="Variance of estimate")
    ci_lower: float = Field(..., description="Lower bound of confidence interval")
    ci_upper: float = Field(..., description="Upper bound of confidence interval")
    contains_true: bool = Field(..., description="Whether CI contains true value")
    ability_id: Optional[str] = Field(None, description="ID of the ability forecast used")
    cost_id: Optional[str] = Field(None, description="ID of the cost trend used")
    design_id: Optional[str] = Field(None, description="ID of the evaluation design")
    budget_fraction: Optional[float] = Field(None, description="Budget as fraction of gold standard")
    ability_variant: Optional[str] = Field(None, description="Variant of ability model (base, lower, upper)")
    cost_variant: Optional[str] = Field(None, description="Variant of cost model (base, lower, upper)")
    base_ability_id: Optional[str] = Field(None, description="Base ability ID without variant suffix")
    base_cost_id: Optional[str] = Field(None, description="Base cost ID without variant suffix")
    window_lower: float = Field(..., description="Lower bound of evaluation window")
    window_upper: float = Field(..., description="Upper bound of evaluation window")
    original_window_lower: float = Field(..., description="Original lower bound of evaluation window")
    original_window_upper: float = Field(..., description="Original upper bound of evaluation window")
    total_samples: int = Field(..., description="Total number of tasks to sample")
