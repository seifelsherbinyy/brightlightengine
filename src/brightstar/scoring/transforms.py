"""Transform functions for metric standardization and normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal, Optional


def parse_percent_strings(series: pd.Series) -> pd.Series:
    """
    Parse percentage strings (e.g., '43.5%') to float values in [0, 1].
    
    Args:
        series: Input series that may contain percent strings
        
    Returns:
        Series with percent strings converted to floats
    """
    if series.dtype == 'object':
        def parse_value(val):
            if pd.isna(val):
                return val
            if isinstance(val, str) and '%' in val:
                try:
                    # Strip % and convert to float, divide by 100
                    numeric = float(val.strip().rstrip('%'))
                    return numeric / 100.0
                except (ValueError, AttributeError):
                    return val
            return val
        return series.apply(parse_value)
    return series


def winsorize(series: pd.Series, q_low: float, q_high: float) -> pd.Series:
    """
    Clip series values at specified quantiles (winsorization).
    
    Args:
        series: Input numeric series
        q_low: Lower quantile (e.g., 0.01 for 1st percentile)
        q_high: Upper quantile (e.g., 0.99 for 99th percentile)
        
    Returns:
        Winsorized series with values clipped at quantiles
    """
    if series.empty or series.isna().all():
        return series
    
    lower_bound = series.quantile(q_low)
    upper_bound = series.quantile(q_high)
    return series.clip(lower=lower_bound, upper=upper_bound)


def apply_direction(series: pd.Series, direction: str) -> pd.Series:
    """
    Apply direction transformation to series.
    For 'lower_is_better', multiply by -1 so higher values are worse.
    
    Args:
        series: Input numeric series
        direction: 'higher_is_better' or 'lower_is_better'
        
    Returns:
        Transformed series
    """
    direction_lower = str(direction).lower()
    if direction_lower in ['lower_is_better', 'down', 'lower', 'desc', 'descending']:
        return -1 * series
    return series


def log1p_safe(series: pd.Series) -> pd.Series:
    """
    Apply log1p transformation with safety clipping at 0.
    
    Args:
        series: Input numeric series
        
    Returns:
        Log1p transformed series
    """
    clipped = series.clip(lower=0)
    return np.log1p(clipped)


def logit_safe(series: pd.Series) -> pd.Series:
    """
    Apply logit transformation with safety clipping to (1e-5, 1-1e-5).
    logit(p) = log(p / (1 - p))
    
    Args:
        series: Input series with values in [0, 1]
        
    Returns:
        Logit transformed series
    """
    # First ensure values are in [0, 1] range
    clipped = series.clip(lower=0, upper=1)
    # Then apply safety margins to avoid log(0) or log(inf)
    eps = 1e-5
    clipped = clipped.clip(lower=eps, upper=1-eps)
    # Apply logit transformation
    return np.log(clipped / (1 - clipped))


def standardize_percentile(
    df: pd.DataFrame,
    value_col: str,
    by_keys: Optional[list[str]] = None
) -> pd.Series:
    """
    Standardize using percentile rank per group, then map to z-like scale.
    
    Process:
    1. Compute percentile rank per group (0 to 1)
    2. Clip to (1e-3, 1-1e-3) to avoid infinities
    3. Apply inverse normal CDF to get z-like scores
    
    Args:
        df: DataFrame containing the data
        value_col: Name of column to standardize
        by_keys: Optional list of columns to group by
        
    Returns:
        Series with standardized values
    """
    if by_keys:
        # Rank within groups
        ranks = df.groupby(by_keys)[value_col].rank(pct=True)
    else:
        # Rank across entire series
        ranks = df[value_col].rank(pct=True)
    
    # Clip to avoid infinities when applying inverse normal
    eps = 1e-3
    ranks_clipped = ranks.clip(lower=eps, upper=1-eps)
    
    # Apply inverse normal CDF (ppf = percent point function)
    z_scores = ranks_clipped.apply(lambda p: stats.norm.ppf(p))
    
    return z_scores


def standardize_robust_z(
    df: pd.DataFrame,
    value_col: str,
    by_keys: Optional[list[str]] = None,
    iqr_floor: float = 1e-6
) -> pd.Series:
    """
    Standardize using robust z-score: (x - median) / IQR.
    
    Args:
        df: DataFrame containing the data
        value_col: Name of column to standardize
        by_keys: Optional list of columns to group by
        iqr_floor: Minimum IQR value to avoid division by zero
        
    Returns:
        Series with robust z-scores
    """
    if by_keys:
        grouped = df.groupby(by_keys)[value_col]
        median = grouped.transform('median')
        q25 = grouped.transform(lambda x: x.quantile(0.25))
        q75 = grouped.transform(lambda x: x.quantile(0.75))
    else:
        median = df[value_col].median()
        q25 = df[value_col].quantile(0.25)
        q75 = df[value_col].quantile(0.75)
    
    iqr = q75 - q25
    # Apply floor to IQR to avoid division by zero
    iqr = np.maximum(iqr, iqr_floor)
    
    robust_z = (df[value_col] - median) / iqr
    return robust_z


def impute(
    df: pd.DataFrame,
    value_col: str,
    policy: Literal['drop', 'group_median', 'zero'],
    group_keys: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Apply imputation strategy for missing values.
    
    Args:
        df: DataFrame containing the data
        value_col: Name of column to impute
        policy: Imputation strategy - 'drop', 'group_median', or 'zero'
        group_keys: Optional list of columns to group by for group_median
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    if policy == 'drop':
        # Drop rows with missing values
        df = df.dropna(subset=[value_col])
    elif policy == 'group_median':
        # Impute with group median
        if group_keys:
            df[value_col] = df.groupby(group_keys)[value_col].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            # Fallback to overall median
            df[value_col] = df[value_col].fillna(df[value_col].median())
    elif policy == 'zero':
        # Impute with zero
        df[value_col] = df[value_col].fillna(0)
    else:
        raise ValueError(f"Unknown imputation policy: {policy}")
    
    return df
