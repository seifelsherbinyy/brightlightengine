"""Unit tests for scoring transforms."""
import numpy as np
import pandas as pd
import pytest

from brightstar.scoring.transforms import (
    apply_direction,
    impute,
    log1p_safe,
    logit_safe,
    parse_percent_strings,
    standardize_percentile,
    standardize_robust_z,
    winsorize,
)


def test_parse_percent_strings():
    """Test parsing of percentage strings."""
    # Test with percent strings
    series = pd.Series(['50%', '75.5%', '100%', '0%'])
    result = parse_percent_strings(series)
    expected = pd.Series([0.5, 0.755, 1.0, 0.0])
    pd.testing.assert_series_equal(result, expected)
    
    # Test with mixed types
    series = pd.Series(['50%', 0.25, None, '100%'])
    result = parse_percent_strings(series)
    assert result[0] == 0.5
    assert result[1] == 0.25
    assert pd.isna(result[2])
    assert result[3] == 1.0
    
    # Test with numeric series (should pass through)
    series = pd.Series([0.5, 0.75, 1.0])
    result = parse_percent_strings(series)
    pd.testing.assert_series_equal(result, series)


def test_winsorize():
    """Test winsorization clamps at quantiles."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Winsorize at 10% and 90%
    result = winsorize(series, 0.1, 0.9)
    
    # Values should be clipped
    lower_bound = series.quantile(0.1)
    upper_bound = series.quantile(0.9)
    
    assert result.min() == lower_bound
    assert result.max() == upper_bound
    
    # Check that middle values are unchanged
    assert result.iloc[4] == 5
    assert result.iloc[5] == 6


def test_winsorize_empty_series():
    """Test winsorization with empty series."""
    series = pd.Series([], dtype=float)
    result = winsorize(series, 0.01, 0.99)
    assert result.empty


def test_apply_direction_higher_is_better():
    """Test direction application for higher_is_better."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = apply_direction(series, 'higher_is_better')
    pd.testing.assert_series_equal(result, series)


def test_apply_direction_lower_is_better():
    """Test direction application for lower_is_better."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = apply_direction(series, 'lower_is_better')
    expected = pd.Series([-1, -2, -3, -4, -5])
    pd.testing.assert_series_equal(result, expected)


def test_apply_direction_aliases():
    """Test direction aliases work correctly."""
    series = pd.Series([1, 2, 3])
    
    # Test aliases for lower_is_better
    for direction in ['lower_is_better', 'down', 'lower', 'desc']:
        result = apply_direction(series, direction)
        assert (result == -series).all()


def test_log1p_safe():
    """Test log1p with negative clipping."""
    # Test with positive values
    series = pd.Series([0, 1, 10, 100])
    result = log1p_safe(series)
    expected = np.log1p(series)
    pd.testing.assert_series_equal(result, expected)
    
    # Test with negative values (should be clipped to 0)
    series = pd.Series([-5, -1, 0, 1, 5])
    result = log1p_safe(series)
    # Negative values should become log1p(0) = 0
    assert result.iloc[0] == 0
    assert result.iloc[1] == 0
    assert result.iloc[2] == 0
    assert result.iloc[3] == np.log1p(1)
    assert result.iloc[4] == np.log1p(5)


def test_logit_safe():
    """Test logit with safety clipping."""
    # Test with values in valid range
    series = pd.Series([0.1, 0.5, 0.9])
    result = logit_safe(series)
    
    # Logit should be monotonic
    assert result.iloc[0] < result.iloc[1] < result.iloc[2]
    
    # Test with edge cases (0 and 1)
    series = pd.Series([0, 0.5, 1])
    result = logit_safe(series)
    
    # Should not have infinite values
    assert not np.isinf(result).any()
    
    # Test with out-of-range values (should be clipped)
    series = pd.Series([-0.5, 0.5, 1.5])
    result = logit_safe(series)
    
    # Should be clipped and no infinities
    assert not np.isinf(result).any()


def test_standardize_percentile():
    """Test percentile standardization."""
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'group': ['A'] * 5 + ['B'] * 5
    })
    
    # Test without grouping
    result = standardize_percentile(df, 'value')
    
    # Should be monotonically increasing
    assert (result.diff().dropna() > 0).all()
    
    # Test with grouping
    result = standardize_percentile(df, 'value', by_keys=['group'])
    
    # Result should be numeric and not have NaN
    assert not result.isna().any()
    
    # Within each group, higher values should have higher z-scores
    df['z'] = result
    group_a_z = df[df['group'] == 'A']['z'].values
    group_b_z = df[df['group'] == 'B']['z'].values
    
    # Check monotonicity within groups
    assert (np.diff(group_a_z) > 0).all()
    assert (np.diff(group_b_z) > 0).all()


def test_standardize_robust_z():
    """Test robust z-score standardization."""
    df = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 100],  # 100 is outlier
        'group': ['A'] * 6
    })
    
    # Test without grouping
    result = standardize_robust_z(df, 'value')
    
    # Median should map to approximately 0
    median_idx = 2  # Value 3 is near median
    assert result.iloc[median_idx] == pytest.approx(0, abs=0.5)
    
    # Test with grouping
    df2 = pd.DataFrame({
        'value': [1, 2, 3, 10, 20, 30],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']
    })
    result = standardize_robust_z(df2, 'value', by_keys=['group'])
    
    # Each group should have median near 0
    df2['z'] = result
    assert df2[df2['group'] == 'A']['z'].iloc[1] == pytest.approx(0, abs=0.5)
    assert df2[df2['group'] == 'B']['z'].iloc[1] == pytest.approx(0, abs=0.5)


def test_standardize_robust_z_constant_series():
    """Test robust z-score handles constant series via IQR floor."""
    df = pd.DataFrame({
        'value': [5, 5, 5, 5, 5],
        'group': ['A'] * 5
    })
    
    # Should not raise division by zero
    result = standardize_robust_z(df, 'value')
    
    # All values should be 0 (since all equal median)
    assert (result == 0).all()


def test_impute_drop():
    """Test drop imputation policy."""
    df = pd.DataFrame({
        'value': [1, 2, np.nan, 4, 5],
        'group': ['A', 'A', 'A', 'B', 'B']
    })
    
    result = impute(df, 'value', policy='drop')
    
    # Should have 4 rows (dropped the NaN)
    assert len(result) == 4
    assert not result['value'].isna().any()


def test_impute_zero():
    """Test zero imputation policy."""
    df = pd.DataFrame({
        'value': [1, 2, np.nan, 4, 5],
        'group': ['A', 'A', 'A', 'B', 'B']
    })
    
    result = impute(df, 'value', policy='zero')
    
    # Should have 5 rows, NaN replaced with 0
    assert len(result) == 5
    assert not result['value'].isna().any()
    assert result.iloc[2]['value'] == 0


def test_impute_group_median():
    """Test group median imputation policy."""
    df = pd.DataFrame({
        'value': [1.0, 2.0, np.nan, 10.0, np.nan],
        'group': ['A', 'A', 'A', 'B', 'B']
    })
    
    result = impute(df, 'value', policy='group_median', group_keys=['group'])
    
    # Should have 5 rows
    assert len(result) == 5
    assert not result['value'].isna().any()
    
    # Group A median is 1.5, should be filled
    assert result.iloc[2]['value'] == 1.5
    
    # Group B has only one value, so median is 10
    assert result.iloc[4]['value'] == 10.0


def test_impute_group_median_no_groups():
    """Test group median imputation without groups."""
    df = pd.DataFrame({
        'value': [1.0, 2.0, np.nan, 4.0, 5.0]
    })
    
    result = impute(df, 'value', policy='group_median')
    
    # Should use overall median (3.0)
    assert len(result) == 5
    assert result.iloc[2]['value'] == 3.0
