#!/usr/bin/env python3
"""Choose the four extension values of the chain vocabulary (docs/chain_v1.md); no model is loaded.

Rule: walk EXTENSION_CANDIDATES in recorded order and keep a candidate when, with the values kept so
far, (a) no value is a substring of another and (b) for both pinned tokenizers and every audit prompt
the candidate continuations are valid (spaced forms preserve the prompt tokenization, no cross-value
token collision, prefix-free events). Stop at four.

Usage:
    PYTHONPATH=. python scripts/chain_audit.py --output outputs/followups/chain_v1/vocabulary_audit.json
"""

from __future__ import annotations

import argparse
import json

from scripts.distance_audit import CONFIGS, load_tokenizer
from src.cross_model.chains import DEPTHS, EXTENSION_CANDIDATES, N_EXTENSION, generate_chain
from src.cross_model.protocol import VALUES, write_new
from src.cross_model.tokens import continuations
from src.data.supersession_behavior import _answer_prefix


def audit_prompts(tokenizer):
    # Audit prompts are drawn with the base vocabulary only (deterministic, never scored).
    rows = [r for d in DEPTHS for r in generate_chain('development', 3, d, 20261199, list(VALUES) + ['x1', 'x2', 'x3', 'x4'])
            if not r['edited']]
    return [_answer_prefix(r['prompt'], tokenizer, True) for r in rows[:: max(1, len(rows) // 24)]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    tokenizers = {c.split('/')[-1].removesuffix('.yaml'): load_tokenizer(c) for c in CONFIGS}
    prompts = {name: audit_prompts(tok) for name, tok in tokenizers.items()}
    kept, log = [], []
    for cand in EXTENSION_CANDIDATES:
        if len(kept) == N_EXTENSION:
            break
        vocab = list(VALUES) + kept + [cand]
        if any(x != y and x in y for x in vocab for y in vocab):
            log.append({'candidate': cand, 'kept': False, 'reason': 'substring of another value'})
            continue
        failures = {}
        for name, tok in tokenizers.items():
            for prompt in prompts[name]:
                try:
                    continuations(tok, prompt, vocab)
                except ValueError as exc:
                    failures[name] = str(exc)
                    break
        log.append({'candidate': cand, 'kept': not failures, 'failures': failures})
        if not failures:
            kept.append(cand)
    report = {'rule': __doc__.split('Rule: ')[1].split('\n\n')[0].replace('\n', ' '),
              'audit': log, 'extension': kept, 'vocabulary': list(VALUES) + kept,
              'complete': len(kept) == N_EXTENSION}
    write_new(a.output, report)
    print(json.dumps(report, indent=2))
    if len(kept) != N_EXTENSION:
        raise SystemExit('fewer than four extension values passed the audit')


if __name__ == '__main__':
    main()
