"""stale_decision_v2r: the frozen replication model list."""
import json

import pytest

from src.experiments import stale_decision_v2m as v2m
from src.experiments import stale_decision_v2r as exp

NAMES = ('qwen3_14b', 'gemma3_12b', 'granite31_8b', 'mistral7b', 'falcon3_7b')


def test_frozen_replication_list():
    for name in NAMES:
        exp.check_config(json.load(open(f'configs/stale_decision_v2r/{name}.json')))
    config = json.load(open('configs/stale_decision_v2r/qwen3_14b.json'))
    with pytest.raises(ValueError):
        exp.check_config({**config, 'model': {**config['model'], 'revision': '0' * 40}})
    with pytest.raises(ValueError):
        exp.check_config({**config, 'model': {**config['model'], 'dtype': 'bfloat16'}})
    with pytest.raises(ValueError):  # v2m configs belong to the sealed protocol
        exp.check_config(json.load(open('configs/stale_decision_v2m/qwen3_8b.json')))


def test_replication_does_not_change_sealed_protocol_hashes():
    assert exp.code_hash() != v2m.code_hash()
    assert set(exp.MODELS).isdisjoint(v2m.MODELS)
