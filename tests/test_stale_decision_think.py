"""stale_decision_think: frozen models, item set and strict parsing after the trace."""
import json

import pytest

from src.experiments import stale_decision_think as exp


def test_configs_and_items():
    for name in ('qwen3_8b', 'qwen3_14b', 'ministral3_14b_reasoning'):
        exp.check_config(json.load(open(f'configs/stale_decision_think/{name}.json')))
    config = json.load(open('configs/stale_decision_think/qwen3_8b.json'))
    rows = exp.items(config)
    assert len(rows) == 384 * 9 and {r['task'] for r in rows} == {'routing'}
    half = exp.items({**config, 'histories': 'first_half'})
    hids = list(dict.fromkeys(r['history_id'] for r in rows))[:192]
    assert len(half) == 192 * 9 and {r['history_id'] for r in half} == set(hids)
    with pytest.raises(ValueError):
        exp.check_config({**config, 'max_new_tokens': 512})


def test_answer_is_parsed_after_the_trace_only():
    config = json.load(open('configs/stale_decision_think/qwen3_8b.json'))
    r = next(r for r in exp.items(config) if r['family'] == 'superseded')
    ok = exp.parse(f"<think>\nmaybe {r['wrong_action']}\n</think>\n\n{r['answer']}<|im_end|>", config, r, True)
    assert ok['generated']['classification'] == 'correct'
    assert exp.parse(f"<think>\n{r['answer']}", config, r, False)['generated']['classification'] == 'invalid'
    assert exp.parse(f"<think></think>\n\n{r['answer']} because", config, r, True)['generated']['classification'] == 'invalid'
