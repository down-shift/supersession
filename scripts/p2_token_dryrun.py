#!/usr/bin/env python3
"""CPU-only check that a P2 model's tokenizer passes Experiment 1's candidate validation (no model load)."""
import sys

from transformers import AutoTokenizer

from src.cross_model import robustness_v2 as data
from src.cross_model.tokens import validate as validate_tokens
from src.data.io import read_jsonl
from src.utils import load_config

GATE = ('/home/danya/supersession/outputs/cross_model_relational_v2/'
        'factorial_relation_counterbalanced_runtimefix_20261004/qwen3_8b/migrated/gate.jsonl')
for path in sys.argv[1:]:
    m = load_config(path)['model']
    try:
        tok = AutoTokenizer.from_pretrained(m['tokenizer_id'], revision=m['tokenizer_revision'],
                                            trust_remote_code=m.get('trust_remote_code', False))
        audit = validate_tokens(tok, read_jsonl(GATE), renderer=data.render, positions=data.semantic_positions)
        print(path, 'OK', {v: len(e) for v, e in audit['events'].items()}, flush=True)
    except Exception as exc:  # report and continue with the next model
        print(path, 'FAILED', type(exc).__name__, str(exc)[:300], flush=True)
