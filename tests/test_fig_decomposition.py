"""The paper's decomposition figure: estimate extraction (no plotting, no artifacts needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("fig_decomposition", ROOT / "scripts" / "fig_decomposition.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fig_decomposition"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(sup, other, entity, early_amr, sup_amr, ent_amr):
    return {"R_superseded": sup, "R_other_attribute": other, "R_entity_mention": entity,
            "R_early_unassigned_aligned_minus_reversed": early_amr,
            "R_superseded_aligned_minus_reversed": sup_amr,
            "R_entity_mention_aligned_minus_reversed": ent_amr}


def test_exp1_rows_have_the_documented_sign_conventions():
    fig = _load()
    np = pytest.importorskip("numpy")
    from src.cross_model.robustness_analysis import summary

    rows = [_row(2.0, 1.5, 5.0, 6.0, 0.3, -1.0), _row(1.0, 1.0, 4.0, 7.0, 0.1, -3.0)]
    est = fig.exp1_estimates(rows, summary)
    means = [e["mean"] for e in est]
    # superseded - other attribute; entity mention - superseded; paired order differences as saved.
    assert means == pytest.approx([0.25, 3.0, 6.5, 0.2, -2.0])
    assert all(e["source"] == "Exp. 1" and e["n_histories"] == 2 for e in est)
    for e in est:
        lo, hi = e["ci95"]
        assert lo <= e["mean"] <= hi
    assert np.isfinite([v for e in est for v in e["ci95"]]).all()


def test_sealed_rows_are_read_verbatim_from_the_saved_summaries(tmp_path, monkeypatch):
    fig = _load()
    import json
    monkeypatch.setattr(fig, "EXP2", tmp_path)
    monkeypatch.setattr(fig, "ROOT", tmp_path)
    keys = {pattern: [] for _, _, _, pattern, _ in fig.SEALED_ROWS}
    for _, _, _, pattern, key in fig.SEALED_ROWS:
        keys[pattern].append(key)
    for k, (pattern, names) in enumerate(keys.items()):
        path = tmp_path / pattern.format(slug="m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": {
            n: {"n_histories": 96, "mean": k + j / 10, "ci95_history_bootstrap": [k - 1, k + 1]}
            for j, n in enumerate(names)}}))
    inputs = {}
    est = fig.sealed_estimates("m", inputs)
    assert len(est) == len(fig.SEALED_ROWS) and len(inputs) == len(keys)
    for e, (group, label, source, pattern, key) in zip(est, fig.SEALED_ROWS):
        k = list(keys).index(pattern)
        assert (e["group"], e["label"], e["source"]) == (group, label, source)
        assert e["mean"] == k + keys[pattern].index(key) / 10 and e["ci95"] == [k - 1, k + 1]


def test_rows_are_grouped_by_question():
    fig = _load()
    rows = ([{"group": g, "label": l} for g, l, _ in fig.EXP1_ROWS]
            + [{"group": g, "label": l} for g, l, *_ in fig.SEALED_ROWS])
    groups = [r["group"] for r in fig.ordered(rows)]
    assert set(groups) == set(fig.GROUP_ORDER)
    # Each group is contiguous and in the declared order, so the plot's group headers are well defined.
    seen = []
    for g in groups:
        if not seen or seen[-1] != g:
            assert g not in seen
            seen.append(g)
    assert seen == list(fig.GROUP_ORDER)
    # Experiment 1 rows come first within their group.
    first = {g: next(r["label"] for r in fig.ordered(rows) if r["group"] == g) for g in seen}
    assert first["Relation-status comparison"] == "superseded $-$ other attribute"
