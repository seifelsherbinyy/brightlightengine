"""Penalty evaluation engine for scoring."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd


logger = logging.getLogger("brightstar.scoring.penalties")


def evaluate_penalties(
    df_long_std: pd.DataFrame,
    rules: List[Any]
) -> Tuple[pd.Series, pd.Series]:
    """
    Evaluate penalty rules and return total penalties and breakdown.
    
    Args:
        df_long_std: Long format DataFrame with standardized metrics.
                    Expected columns: entity_id, vendor_code, Week_Order, metric, 
                                     std_value, raw_value, coverage
        rules: List of penalty rule dictionaries with keys:
               - name: Rule name
               - when: Query string for pandas query() or boolean condition
               - apply: Penalty value to apply (negative number)
    
    Returns:
        Tuple of (penalty_total, penalty_breakdown):
        - penalty_total: Series with sum of all penalties per row
        - penalty_breakdown: Series with JSON string of applied penalties per row
    """
    if df_long_std.empty or not rules:
        # Return empty series if no data or no rules
        penalty_total = pd.Series(0.0, index=df_long_std.index)
        penalty_breakdown = pd.Series('[]', index=df_long_std.index)
        return penalty_total, penalty_breakdown
    
    df = df_long_std.copy()
    
    # Initialize penalty tracking
    df['_penalty_total'] = 0.0
    df['_penalty_list'] = [[] for _ in range(len(df))]
    
    # Evaluate each rule
    def _field(r: Any, key: str, default: Any = None) -> Any:
        # Support both dict-like and object-like rules (e.g., Pydantic models)
        if isinstance(r, dict):
            return r.get(key, default)
        return getattr(r, key, default)

    for rule in rules:
        rule_name = _field(rule, 'name', 'unnamed')
        when_condition = _field(rule, 'when', '')
        penalty_value = float(_field(rule, 'apply', 0))
        
        if not when_condition:
            logger.warning(f"Penalty rule '{rule_name}' has no 'when' condition, skipping")
            continue
        
        try:
            # Try to evaluate the condition using pandas query
            # query() is safer but requires proper syntax
            mask = df.eval(when_condition)
            
            # Apply penalty where condition is True
            df.loc[mask, '_penalty_total'] += penalty_value
            
            # Track which rules were applied
            for idx in df[mask].index:
                df.at[idx, '_penalty_list'].append({
                    'rule': rule_name,
                    'penalty': penalty_value
                })
            
            hit_count = mask.sum()
            if hit_count > 0:
                logger.info(f"Penalty rule '{rule_name}' applied {hit_count} times")
                
        except Exception as e:
            logger.error(f"Failed to evaluate penalty rule '{rule_name}': {e}")
            logger.debug(f"Rule condition: {when_condition}")
            continue
    
    # Convert penalty list to JSON strings
    penalty_breakdown = df['_penalty_list'].apply(json.dumps)
    penalty_total = df['_penalty_total']
    
    return penalty_total, penalty_breakdown
