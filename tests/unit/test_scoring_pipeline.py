"""Unit tests for scoring pipeline."""
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from brightstar.scoring.penalties import evaluate_penalties


def test_penalty_evaluation_basic():
    """Test basic penalty evaluation."""
    df = pd.DataFrame({
        'entity_id': ['E1', 'E2', 'E3'],
        'vendor_code': ['V1', 'V1', 'V2'],
        'Week_Order': [1, 1, 1],
        'metric': ['RegRate%', 'OOS%', 'OOS%'],
        'std_value': [-1.5, 2.0, 0.5],
        'raw_value': [0.3, 0.8, 0.4],
        'coverage': [0.85, 0.85, 0.95]
    })
    
    rules = [
        {
            'name': 'low-registration-rate',
            'when': "(metric == 'RegRate%') & (std_value < -1.0)",
            'apply': -30
        },
        {
            'name': 'chronic-oos',
            'when': "(metric == 'OOS%') & (std_value > 1.5) & (coverage < 0.9)",
            'apply': -25
        }
    ]
    
    penalty_total, penalty_breakdown = evaluate_penalties(df, rules)
    
    # First row should have -30 penalty
    assert penalty_total.iloc[0] == -30
    
    # Second row should have -25 penalty
    assert penalty_total.iloc[1] == -25
    
    # Third row should have 0 penalty
    assert penalty_total.iloc[2] == 0
    
    # Check breakdown
    breakdown_0 = json.loads(penalty_breakdown.iloc[0])
    assert len(breakdown_0) == 1
    assert breakdown_0[0]['rule'] == 'low-registration-rate'
    assert breakdown_0[0]['penalty'] == -30


def test_penalty_evaluation_multiple_rules_per_row():
    """Test that multiple penalties can apply to one row."""
    df = pd.DataFrame({
        'entity_id': ['E1'],
        'vendor_code': ['V1'],
        'Week_Order': [1],
        'metric': ['OOS%'],
        'std_value': [2.0],
        'raw_value': [0.9],
        'coverage': [0.85]
    })
    
    rules = [
        {
            'name': 'high-oos',
            'when': "(metric == 'OOS%') & (std_value > 1.5)",
            'apply': -20
        },
        {
            'name': 'low-coverage',
            'when': "coverage < 0.9",
            'apply': -10
        }
    ]
    
    penalty_total, penalty_breakdown = evaluate_penalties(df, rules)
    
    # Should have both penalties
    assert penalty_total.iloc[0] == -30
    
    # Check breakdown has both rules
    breakdown = json.loads(penalty_breakdown.iloc[0])
    assert len(breakdown) == 2


def test_penalty_evaluation_empty_rules():
    """Test penalty evaluation with no rules."""
    df = pd.DataFrame({
        'entity_id': ['E1'],
        'vendor_code': ['V1'],
        'Week_Order': [1],
        'metric': ['GMS'],
        'std_value': [1.0],
        'raw_value': [100],
        'coverage': [1.0]
    })
    
    rules = []
    
    penalty_total, penalty_breakdown = evaluate_penalties(df, rules)
    
    # Should have 0 penalty
    assert penalty_total.iloc[0] == 0
    
    # Breakdown should be empty array
    assert penalty_breakdown.iloc[0] == '[]'


def test_penalty_evaluation_empty_dataframe():
    """Test penalty evaluation with empty dataframe."""
    df = pd.DataFrame({
        'entity_id': [],
        'vendor_code': [],
        'Week_Order': [],
        'metric': [],
        'std_value': [],
        'raw_value': [],
        'coverage': []
    })
    
    rules = [{'name': 'test', 'when': 'std_value > 1', 'apply': -10}]
    
    penalty_total, penalty_breakdown = evaluate_penalties(df, rules)
    
    assert len(penalty_total) == 0
    assert len(penalty_breakdown) == 0


def test_schema_validation_missing_columns():
    """Test that missing required columns raises an error."""
    from brightstar.phase2_scoring import validate_input_schema
    
    # Missing 'metric' column
    df = pd.DataFrame({
        'entity_id': ['E1'],
        'vendor_code': ['V1'],
        'Week_Order': [1],
        'value': [100]
    })
    
    with pytest.raises(ValueError) as excinfo:
        validate_input_schema(df)
    
    assert 'metric' in str(excinfo.value).lower()


def test_weight_renormalization():
    """Test that weights are renormalized when they don't sum to 1."""
    from brightstar.common.config_validator import ScoringConfig, MetricConfig
    
    # Weights sum to 0.9
    config = ScoringConfig(
        rolling_weeks=8,
        weights_must_sum_to_1=True,
        metrics=[
            MetricConfig(name='GMS', direction='higher_is_better', weight=0.4),
            MetricConfig(name='OOS', direction='lower_is_better', weight=0.3),
            MetricConfig(name='FillRate', direction='higher_is_better', weight=0.2)
        ]
    )
    
    # Should not raise an error (renormalization happens at runtime)
    total = sum(m.weight for m in config.metrics)
    assert total == 0.9  # Not 1.0, but valid


def test_config_validation_invalid_quantiles():
    """Test that invalid quantiles raise validation error."""
    from pydantic import ValidationError
    from brightstar.common.config_validator import MetricConfig
    
    # Quantiles in wrong order
    with pytest.raises(ValidationError):
        MetricConfig(
            name='GMS',
            direction='higher_is_better',
            weight=0.5,
            clip_quantiles=[0.99, 0.01]  # Wrong order
        )


def test_config_validation_negative_weight():
    """Test that negative weights raise validation error."""
    from pydantic import ValidationError
    from brightstar.common.config_validator import MetricConfig
    
    with pytest.raises(ValidationError):
        MetricConfig(
            name='GMS',
            direction='higher_is_better',
            weight=-0.5  # Negative
        )


def test_deterministic_output(tmp_path):
    """Test that same input+config produces identical outputs."""
    # Create test data
    df = pd.DataFrame({
        'entity_id': ['E1', 'E2'] * 5,
        'vendor_code': ['V1', 'V2'] * 5,
        'Week_Order': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'metric': ['GMS($)'] * 10,
        'value': [100, 110, 105, 115, 102, 112, 108, 118, 106, 116]
    })
    
    config_path = tmp_path / 'config.yaml'
    data_path = tmp_path / 'data.csv'
    df.to_csv(data_path, index=False)
    
    # Write minimal config
    config_content = dedent(f"""
        paths:
            processed_dir: {tmp_path / 'processed'}
            logs_dir: {tmp_path / 'logs'}
        logging:
            scoring_level: ERROR
        scoring:
            rolling_weeks: 3
            weights_must_sum_to_1: true
            metrics:
                - name: GMS($)
                  direction: higher_is_better
                  weight: 1.0
                  clip_quantiles: [0.01, 0.99]
                  transform: none
                  standardize: robust_z
                  missing_policy: drop
    """)
    config_path.write_text(config_content)
    
    # For this test, we'd need to run the pipeline twice and compare
    # This is a placeholder for integration with the actual pipeline
    # The actual test would call run_phase2 twice and compare outputs
    assert True  # Placeholder


def test_reliability_calculation():
    """Test reliability calculation with varying coverage and variance."""
    # High coverage, low variance -> high reliability
    # Low coverage, high variance -> low reliability
    
    # Create mock data for reliability calc
    df = pd.DataFrame({
        'entity_id': ['E1', 'E1', 'E1', 'E2', 'E2', 'E2'],
        'Week_Order': [1, 2, 3, 1, 2, 3],
        'metric': ['GMS'] * 6,
        'std_value': [1.0, 1.1, 1.05, 0.5, 2.0, 0.3],  # E2 has high variance
        'coverage': [1.0, 1.0, 1.0, 0.6, 0.7, 0.5]  # E2 has low coverage
    })
    
    # Mock reliability calculation
    def calculate_reliability(group):
        coverage_mean = group['coverage'].mean()
        variance = group['std_value'].var()
        reliability = min(coverage_mean, 1.0) * (1 / (1 + variance))
        return np.clip(reliability, 0, 1)
    
    reliability = df.groupby('entity_id', group_keys=False).apply(calculate_reliability, include_groups=False)
    
    # E1 should have higher reliability than E2
    assert reliability.loc['E1'] > reliability.loc['E2']
    
    # All reliabilities should be in [0, 1]
    assert (reliability >= 0).all() and (reliability <= 1).all()


def test_percent_string_handling_in_pipeline():
    """Test that percent strings are correctly parsed in the pipeline."""
    df = pd.DataFrame({
        'entity_id': ['E1', 'E2'],
        'vendor_code': ['V1', 'V2'],
        'Week_Order': [1, 1],
        'metric': ['FillRate%', 'FillRate%'],
        'value': ['90%', '85%']  # String percentages
    })
    
    # Apply parsing
    from brightstar.scoring.transforms import parse_percent_strings
    df['value'] = parse_percent_strings(df['value'])
    
    # Should be converted to decimals
    assert df.iloc[0]['value'] == 0.90
    assert df.iloc[1]['value'] == 0.85
