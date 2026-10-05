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
    assert all(e["experiment"] == 1 and e["n_histories"] == 2 for e in est)
    for e in est:
        lo, hi = e["ci95"]
        assert lo <= e["mean"] <= hi
    assert np.isfinite([v for e in est for v in e["ci95"]]).all()


def test_exp2_rows_are_read_verbatim_from_the_sealed_summary():
    fig = _load()
    analysis = {"summary": {
        "superseded_marker_effect_all": {"n_histories": 24, "mean": -0.75, "ci95_history_bootstrap": [-0.9, -0.6]},
        "entity_mention_marker_effect_all": {"n_histories": 24, "mean": -2.0, "ci95_history_bootstrap": [-2.4, -1.7]},
    }}
    est = fig.exp2_estimates(analysis)
    assert [(e["label"], e["mean"], e["ci95"]) for e in est] == [
        ("superseded", -0.75, [-0.9, -0.6]), ("entity mention", -2.0, [-2.4, -1.7])]
    assert all(e["experiment"] == 2 for e in est)


def test_rows_from_both_experiments_share_one_group_order():
    fig = _load()
    groups = [g for g, _, _ in fig.EXP1_ROWS] + [g for g, _, _ in fig.EXP2_ROWS]
    # Each group is contiguous, so the plot's group headers are well defined.
    seen = []
    for g in groups:
        if not seen or seen[-1] != g:
            assert g not in seen
            seen.append(g)
