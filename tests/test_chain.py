"""Update-chain experiment: generation, validation and analysis (no model needed)."""

from __future__ import annotations

import pytest

from src.cross_model.chains import (DEPTHS, EXTENSION_CANDIDATES, analyze_chain, chain_vocabulary,
                                    generate_chain)

VOCAB = chain_vocabulary(EXTENSION_CANDIDATES[:4])


@pytest.mark.parametrize('depth', DEPTHS)
def test_chain_prompts_reveal_disjoint_chains_and_edit_the_superseded_value(depth):
    rows = generate_chain('development', 6, depth, 1, VOCAB)
    assert len(rows) == 6 * 32
    r = next(x for x in rows if not x['edited'] and x['historical_order'] == 0 and x['current_order'] == 0)
    lines = r['prompt'].split('\n')
    assert len(lines) == 2 * depth + 2
    assert lines[0].startswith('Initially, ') and lines[2 * depth - 1].startswith('Currently, ')
    e = r['entities'][0]
    assert set(r['chains'][e]).isdisjoint(r['chains'][r['entities'][1]])
    assert r['historical_values'][e] == r['chains'][e][-2]


def test_chain_vocabulary_rules():
    with pytest.raises(ValueError):
        chain_vocabulary(['olive', 'plum', 'navy'])
    with pytest.raises(ValueError):
        chain_vocabulary(['olive', 'plum', 'navy', 'amberx'])  # 'amber' is a substring


def test_chain_analysis_counts_answers_and_donor_following():
    rows = generate_chain('development', 4, 3, 2, VOCAB)
    scores = []
    for r in rows:
        masses = {v: -10.0 for v in VOCAB}
        matched = r['edited_variable'] == r['query_variable']
        # The model answers with the donor exactly when the edited entity is queried, else correctly.
        parsed = r['replacement_value'] if (r['edited'] and matched) else r['answer']
        if r['edited'] and matched:
            masses[r['replacement_value']] += 2.0
        scores.append({**r, 'semantic_log_mass': masses, 'parsed_answer': parsed})
    s = analyze_chain(rows, scores)['summary']
    assert s['donor_matched']['mean'] == pytest.approx(1.0)
    assert s['donor_other']['mean'] == pytest.approx(0.0)
    assert s['donor_following']['mean'] == pytest.approx(1.0)
    assert s['correct']['mean'] == pytest.approx(0.75)
    assert s['stale']['mean'] == pytest.approx(0.25)
    assert s['R']['mean'] == pytest.approx(2.0)
