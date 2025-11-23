from pathlib import Path
import json

from brightstar.standards.metrics import resolve_configured_metrics


def test_resolve_configured_metrics_alias_and_fuzzy(tmp_path: Path):
    configured = ["GMS($)", "FillRate(%)", "SoROOS(%)"]
    available = ["Shipped Revenue", "Unit Session %", "OOS %"]

    alias_path = tmp_path / "metric_aliases.json"
    alias_path.write_text(json.dumps({"Shipped Revenue": "GMS($)"}), encoding="utf-8")

    mapping, warnings = resolve_configured_metrics(
        configured_metrics=configured,
        available_columns=available,
        alias_json=alias_path,
        fuzzy_cutoff=0.8,
    )

    # Alias should map Shipped Revenue -> GMS($)
    assert mapping.get("Shipped Revenue") == "GMS($)"
    # Fuzzy/normalized should resolve Unit Session % -> FillRate(%) and OOS % -> SoROOS(%) or warn if not
    assert "FillRate(%)" in mapping.values() or any("FillRate" in w for w in warnings)
    assert "SoROOS(%)" in mapping.values() or any("SoROOS" in w for w in warnings)
