"""stale_decision_v2f: frozen model list, load classes and tokenizer options."""
import json

import pytest

from src.experiments import stale_decision_v2f as exp
from src.experiments import stale_decision_v2r as v2r

NAMES = ('ministral3_14b', 'gemma4_12b', 'qwen35_9b', 'granite41_8b', 'olmo3_7b')


def test_frozen_family_list():
    for name in NAMES:
        exp.check_config(json.load(open(f'configs/stale_decision_v2f/{name}.json')))
    config = json.load(open('configs/stale_decision_v2f/ministral3_14b.json'))
    assert config['model']['tokenizer_kwargs'] == {'fix_mistral_regex': True}
    for bad in ({'revision': '0' * 40}, {'dtype': 'bfloat16'}, {'load_class': 'vision'}, {'tokenizer_kwargs': None}):
        with pytest.raises(ValueError):
            exp.check_config({**config, 'model': {**config['model'], **bad}})
    with pytest.raises(ValueError):  # other protocols' configs are rejected
        exp.check_config(json.load(open('configs/stale_decision_v2r/qwen3_14b.json')))


def test_family_module_keeps_sealed_hashes_separate():
    assert exp.code_hash() != v2r.code_hash()
    assert set(exp.MODELS).isdisjoint(v2r.MODELS)
