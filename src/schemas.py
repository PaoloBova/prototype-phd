"""
Pydantic models for the forecast detection case study.
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from datetime import datetime

class RawDataRecord(BaseModel):
    """Raw record from detection dataset."""
    task_id: str = Field(..., description="Unique identifier for the task")
    model: str = Field(..., description="Model used for generation")
    alias: Optional[str] = Field(None, description="Alternative name for model")
    task_source: str = Field(..., description="Source of the task")
    human_minutes: float = Field(..., description="Human time to complete task in minutes")
    generation_cost: float = Field(..., description="Cost of generation")
    score_binarized: int = Field(..., description="Binary success indicator (1=success, 0=failure)")
    
    class Config:
        arbitrary_types_allowed = True

class ProcessedDataRecord(RawDataRecord):
    """Processed record with derived features."""
    human_seconds: float = Field(..., description="Human time in seconds")
    log2_human_seconds: float = Field(..., description="Log2 of human time in seconds")
    bin_power: int = Field(..., description="Discretized bin based on log2 scale")
    date: Optional[datetime] = Field(None, description="Release date of the model")
    
    class Config:
        arbitrary_types_allowed = True

class LogisticFitParams(BaseModel):
    """Parameters for a logistic curve fit."""
    threshold: float = Field(..., description="Threshold parameter (inflection point)")
    slope: float = Field(..., description="Slope parameter (steepness)")
    model: str = Field(..., description="Model identifier")
    date: datetime = Field(..., description="Date of the fitted data")
    
    class Config:
        arbitrary_types_allowed = True

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
    
    class Config:
        arbitrary_types_allowed = True

class CostTrend(BaseModel):
    """Cost doubling trend."""
    doubling_rate: float = Field(..., description="Task difficulty units per doubling of cost")
    intercept: float = Field(..., description="Base cost at difficulty 0")
    r_squared: float = Field(..., description="Goodness of fit")
    
    class Config:
        arbitrary_types_allowed = True

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
    
    class Config:
        arbitrary_types_allowed = True

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
    
    class Config:
        arbitrary_types_allowed = True

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

class TaskSampler(BaseModel):
    """Base model for task sampling strategy."""
    sampler_type: TaskSamplerType = Field(..., description="Type of sampling distribution")
    
    class Config:
        arbitrary_types_allowed = True

class UniformTaskSampler(TaskSampler):
    """Uniform distribution sampler for tasks."""
    sampler_type: TaskSamplerType = Field(TaskSamplerType.UNIFORM, literal=True)
    
    def sample(self, n_samples: int, window_lower: float, window_upper: float) -> np.ndarray:
        """Sample n_samples tasks uniformly from window."""
        return np.random.uniform(window_lower, window_upper, n_samples)

class NormalTaskSampler(TaskSampler):
    """Normal distribution sampler for tasks."""
    sampler_type: TaskSamplerType = Field(TaskSamplerType.NORMAL, literal=True)
    mean_offset: float = Field(0.0, description="Offset from center of window for mean")
    std_dev_factor: float = Field(0.3, description="Factor of window width for standard deviation")
    
    def sample(self, n_samples: int, window_lower: float, window_upper: float) -> np.ndarray:
        """Sample n_samples tasks from a normal distribution, clipped to window."""
        window_width = window_upper - window_lower
        window_center = (window_lower + window_upper) / 2
        mean = window_center + (window_width * self.mean_offset)
        std = window_width * self.std_dev_factor
        samples = np.random.normal(mean, std, n_samples)
        # Truncate to window bounds
        return np.clip(samples, window_lower, window_upper)

class EvaluationDesign(BaseModel):
    """Parameters defining how evaluations are designed and sampled."""
    sampler_type: TaskSamplerType = Field(TaskSamplerType.UNIFORM, description="Type of task distribution")
    adjustment_method: WindowAdjustmentMethod = Field(
        WindowAdjustmentMethod.UPPER_BOUND, 
        description="Method used to adjust window based on budget"
    )
    repeats_per_unit: int = Field(20, description="Number of sample repeats per difficulty unit")
    sampler_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the sampler")

class EvaluationScenario(BaseModel):
    """A scenario for evaluation combining ability, cost, and resource constraints."""
    ability: AbilityForecast = Field(..., description="Ability forecast for this scenario")
    doubling_rate: float = Field(..., description="Cost doubling rate in difficulty units")
    budget_fraction: float = Field(..., description="Budget as fraction of gold standard")
    scenario_id: str = Field(..., description="Unique identifier for this scenario")
    ability_id: str = Field(..., description="ID of the ability forecast used")
    cost_id: str = Field(..., description="ID of the cost trend used")
    constraint_id: str = Field(..., description="ID of the resource constraint applied")
    cost_model: str = Field(..., description="Name of the cost model used")
    
    class Config:
        arbitrary_types_allowed = True

class EvaluationForecast(BaseModel):
    """Parameters defining an evaluation forecast under resource constraints."""
    ability: AbilityForecast = Field(..., description="Ability forecast for this evaluation")
    budget_fraction: float = Field(..., description="Budget as fraction of gold standard")
    budget_scenario: str = Field(..., description="Budget scenario name")
    window_lower: float = Field(..., description="Lower bound of evaluation window")
    window_upper: float = Field(..., description="Upper bound of evaluation window")
    sampler_type: TaskSamplerType = Field(TaskSamplerType.UNIFORM, description="Type of task distribution")
    total_samples: int = Field(..., description="Total number of tasks to sample")
    gold_standard_cost: float = Field(..., description="Total cost of gold standard evaluation")
    available_budget: float = Field(..., description="Available budget (gold_standard_cost * budget_fraction)")
    adjustment_method: WindowAdjustmentMethod = Field(
        WindowAdjustmentMethod.UPPER_BOUND, 
        description="Method used to adjust window based on budget"
    )
    original_window_lower: float = Field(..., description="Original lower bound of evaluation window")
    original_window_upper: float = Field(..., description="Original upper bound of evaluation window")
    # Include design parameters
    repeats_per_unit: int = Field(20, description="Number of sample repeats per difficulty unit")
    # Include cost model info
    cost_model: str = Field(..., description="Name of the cost model used")
    doubling_rate: float = Field(..., description="Cost doubling rate in difficulty units")
    ability_id: str = Field(..., description="ID of the ability forecast used")
    cost_id: str = Field(..., description="ID of the cost trend used")
    constraint_id: str = Field(..., description="ID of the resource constraint applied")
    design_id: str = Field(..., description="Unique identifier for the evaluation design method")
    # Add new fields for variant information
    ability_variant: str = Field("unknown", description="Variant of ability model (base, lower, upper)")
    cost_variant: str = Field("unknown", description="Variant of cost model (base, lower, upper)")
    base_ability_id: str = Field("", description="Base ID of ability forecast without variant suffix")
    base_cost_id: str = Field("", description="Base ID of cost trend without variant suffix")
    
    class Config:
        arbitrary_types_allowed = True

class BootstrapConfig(BaseModel):
    """Configuration for bootstrap analysis."""
    n_bootstrap: int = Field(1000, description="Number of bootstrap samples")
    sample_size: Optional[int] = Field(None, description="Size of each bootstrap sample")
    weights: Optional[List[float]] = Field(None, description="Sampling weights")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    class Config:
        arbitrary_types_allowed = True

