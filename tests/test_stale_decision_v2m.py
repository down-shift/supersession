"""stale_decision_v2m: end-of-turn token handling and the frozen model list."""
import json

import pytest

from src.experiments import stale_decision_v2m as exp
from tests.test_stale_decision_v1 import Tokenizer


class TurnTokenizer(Tokenizer):
    unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return {'<end_of_turn>': 106}.get(token, self.unk_token_id)


def test_end_of_turn_replaces_eos_everywhere():
    tok = exp.EndOfTurn(TurnTokenizer(), '<end_of_turn>')
    assert tok.eos_token_id == 106 and tok.chat_template == 'test'
    events = exp.action_events(tok, 'prompt', ['APPROVE', 'DENY'])
    assert events['APPROVE'][0]['ids'][-1] == 106
    assert tok('ab')['input_ids'] == [97, 98]
    with pytest.raises(ValueError):
        exp.EndOfTurn(TurnTokenizer(), '<nope>')


def test_frozen_model_list():
    for name in ('qwen3_8b', 'gemma3_4b'):
        exp.check_config(json.load(open(f'configs/stale_decision_v2m/{name}.json')))
    config = json.load(open('configs/stale_decision_v2m/gemma3_4b.json'))
    with pytest.raises(ValueError):
        exp.check_config({**config, 'model': {**config['model'], 'revision': '0' * 40}})
    with pytest.raises(ValueError):
        exp.check_config({**config, 'model': {k: v for k, v in config['model'].items() if k != 'end_token'}})


def test_end_suffix_is_scored_before_the_end_token():
    tok = exp.EndOfTurn(TurnTokenizer(), '<end_of_turn>', '\n')
    ids = exp.action_events(tok, 'prompt', ['DENY', 'APPROVE'])['DENY'][0]['ids']
    assert ids == [ord(c) for c in 'DENY\n'] + [106]
