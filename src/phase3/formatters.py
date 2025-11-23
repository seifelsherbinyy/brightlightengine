from __future__ import annotations

from typing import Any, Dict, Tuple


def _pct(x: Any) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "n/a"


def format_setup_block(event: Dict[str, Any]) -> str:
    metric = event.get("metric") or event.get("dimension") or event.get("signal_type", "")
    vendor = event.get("vendor_id", "?")
    asin = event.get("asin_id")
    week = event.get("canonical_week_id")
    delta = event.get("delta_pct")
    sev = event.get("severity")
    id_part = f"Vendor {vendor}" + (f", ASIN {asin}" if asin else "")
    return f"{id_part} in {week}: {metric} moved {_pct(delta)} WoW (severity {sev:.2f})."


def format_driver_block(event: Dict[str, Any]) -> str:
    st = event.get("signal_type", "")
    if st.startswith("metric_movement:"):
        metric = st.split(":", 1)[1]
        return f"Primary driver appears to be a week-over-week change in {metric}, aligned with dimension shifts where present."
    if st.startswith("dimension_shift:"):
        dim = st.split(":", 1)[1]
        return f"Key driver is a shift in the {dim} dimension score, reflecting underlying metric movements."
    if st.startswith("inventory_risk:"):
        risk = st.split(":", 1)[1]
        return f"Inventory risk flagged: {risk}. This can depress availability and growth."
    if st.startswith("buyability_failure:"):
        issue = st.split(":", 1)[1]
        return f"Buyability failure detected: {issue}. Listing health is blocking demand capture."
    if st.startswith("readiness_low_completeness"):
        return "Readiness is below threshold; metrics may be unstable due to missing or null-heavy inputs."
    return "Signal detected by scoring engine; cross-verify with dimension and metric deltas."


def format_next_steps_block(event: Dict[str, Any]) -> str:
    st = event.get("signal_type", "")
    steps = []
    # Generic, deadline-oriented guidance
    if st.startswith("buyability_failure:"):
        steps = [
            "Open a ticket to restore buyability (P0) and confirm status within 24h",
            "Validate offer/inventory sync and re-ingest catalog within 48h",
            "Monitor conversion and FO% post-fix next week",
        ]
    elif st.startswith("inventory_risk:"):
        steps = [
            "Align on OTB vs demand; submit PO corrections within 72h",
            "Pull forward replenishment or rebalance from overstock within 1 week",
            "Set guardrails for FO% and instock targets for next 2 weeks",
        ]
    elif st.startswith("dimension_shift:"):
        steps = [
            "Deep-dive the component metrics and confirm anomalies within 48h",
            "Align with vendor on root cause and mitigation by next WBR",
            "Track recovery trajectory week-over-week",
        ]
    else:
        steps = [
            "Validate data inputs and confirm metric integrity within 24h",
            "Align owner for corrective actions and target by next WBR",
            "Review outcomes and update plan in the next business review",
        ]
    return " Next Steps: " + "; ".join(steps) + "."


def compose_wbr_callout(event: Dict[str, Any], config: Dict[str, Any] | None = None) -> Tuple[str, str, str]:
    setup = format_setup_block(event)
    driver = format_driver_block(event)
    next_steps = format_next_steps_block(event)
    return setup, driver, next_steps
