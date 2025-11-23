"""Configuration validation models using Pydantic."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MetricConfig(BaseModel):
    """Configuration for a single metric."""
    
    name: str = Field(..., description="Metric name (e.g., 'GMS($)')")
    direction: Literal['higher_is_better', 'lower_is_better'] = Field(
        ..., description="Whether higher or lower values are better"
    )
    weight: float = Field(..., ge=0, description="Metric weight in composite score")
    clip_quantiles: List[float] = Field(
        [0.01, 0.99],
        description="Winsorization quantiles [low, high]"
    )
    transform: Literal['none', 'log1p', 'logit'] = Field(
        'none',
        description="Transform to apply before standardization"
    )
    standardize: Literal['percentile', 'robust_z'] = Field(
        'percentile',
        description="Standardization method"
    )
    missing_policy: Literal['drop', 'group_median', 'zero'] = Field(
        'group_median',
        description="How to handle missing values"
    )
    
    @field_validator('clip_quantiles')
    @classmethod
    def validate_quantiles(cls, v):
        """Ensure quantiles are valid."""
        if len(v) != 2:
            raise ValueError("clip_quantiles must have exactly 2 values [low, high]")
        if not (0 <= v[0] < v[1] <= 1):
            raise ValueError("clip_quantiles must be in range [0,1] with low < high")
        return v


class PenaltyRule(BaseModel):
    """Configuration for a penalty rule."""
    
    name: str = Field(..., description="Rule name for auditing")
    when: str = Field(..., description="Pandas query condition")
    apply: float = Field(..., description="Penalty to apply (typically negative)")


class CapConfig(BaseModel):
    """Score caps configuration."""
    
    min_score: float = Field(-200, description="Minimum allowed score")
    max_score: float = Field(1000, description="Maximum allowed score")


class ScoringConfig(BaseModel):
    """Complete scoring configuration."""
    
    rolling_weeks: int = Field(8, ge=1, description="Rolling window size for standardization")
    weights_must_sum_to_1: bool = Field(
        True,
        description="Whether to enforce/normalize weights to sum to 1"
    )
    metrics: List[MetricConfig] = Field(..., min_length=1, description="Metric configurations")
    penalties: List[PenaltyRule] = Field(default_factory=list, description="Penalty rules")
    caps: CapConfig = Field(default_factory=CapConfig, description="Score limits")
    
    @model_validator(mode='after')
    def validate_weights(self):
        """Validate that weights are sensible."""
        total_weight = sum(m.weight for m in self.metrics)
        
        if total_weight <= 0:
            raise ValueError("Total metric weights must be positive")
        
        if self.weights_must_sum_to_1:
            # Check if weights sum to approximately 1 (within tolerance)
            if not (0.99 <= total_weight <= 1.01):
                # Will be normalized at runtime with warning
                pass
        
        return self


def load_and_validate_config(config_dict: dict) -> ScoringConfig:
    """
    Load and validate scoring configuration.
    
    Handles both legacy and new config formats for backward compatibility.
    
    Args:
        config_dict: Dictionary with scoring configuration
        
    Returns:
        Validated ScoringConfig object
        
    Raises:
        ValidationError: If configuration is invalid
    """
    scoring_dict = config_dict.get('scoring', {})
    
    # Check if this is a legacy config (has metric_weights but not metrics list)
    if 'metrics' not in scoring_dict and 'metric_weights' in scoring_dict:
        # Convert legacy format to new format
        metrics = []
        weights = scoring_dict.get('metric_weights', {})
        directions = scoring_dict.get('metric_directions', {})
        
        for name, weight in weights.items():
            direction_raw = directions.get(name, 'up')
            # Convert old direction format to new
            if direction_raw in ['up', 'higher']:
                direction = 'higher_is_better'
            elif direction_raw in ['down', 'lower']:
                direction = 'lower_is_better'
            else:
                direction = 'higher_is_better'
            
            metrics.append({
                'name': name,
                'weight': weight,
                'direction': direction,
                'clip_quantiles': [0.01, 0.99],
                'transform': 'none',
                'standardize': 'percentile',
                'missing_policy': 'group_median'
            })
        
        scoring_dict = scoring_dict.copy()
        scoring_dict['metrics'] = metrics
        
        # Add defaults for new fields
        if 'rolling_weeks' not in scoring_dict:
            scoring_dict['rolling_weeks'] = 8
        if 'weights_must_sum_to_1' not in scoring_dict:
            scoring_dict['weights_must_sum_to_1'] = True
    
    return ScoringConfig(**scoring_dict)
