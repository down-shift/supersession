#!/usr/bin/env python3
"""Tokenizer audit for the name-value distance experiment (docs/distance_v1.md); no model is loaded.

Rule (recorded in docs/distance_v1.md before scoring): take the near superseded candidates in their
recorded order; a candidate passes when, for every baseline/edit pair of a 96-history dataset and for
both pinned tokenizers, the chat-rendered token sequences share every token outside the edited value
(the differing middle segment decodes to the source and donor values). The first passing candidate is
used. For every construction x distance cell the audit also records the measured name-to-value
distance: the number of tokens strictly between the last token of the entity name and the first token
of the value, in the historical line.

Usage:
    PYTHONPATH=. python scripts/distance_audit.py --output outputs/followups/distance_v1/tokenizer_audit.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.cross_model.followups import SUPERSEDED_NEAR_CANDIDATES, generate_distance
from src.cross_model.protocol import write_new
from src.cross_model.tokens import encode
from src.data.supersession_behavior import _answer_prefix
from src.utils import load_config

CONFIGS = ('configs/cross_model_relational_v2/qwen3_8b.yaml', 'configs/cross_model_relational_v2/gemma3_4b.yaml')


def load_tokenizer(config_path):
    from transformers import AutoTokenizer
    m = load_config(config_path)['model']
    return AutoTokenizer.from_pretrained(m.get('tokenizer_id', m['id']),
                                         revision=m.get('tokenizer_revision', m['revision']))


def middle(a, b):
    """Return (prefix length, a-middle, b-middle) after stripping the longest common prefix/suffix."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return i, a[i:len(a) - j], b[i:len(b) - j]


def span_ok(tokenizer, base, edit):
    """True when the base/edit token sequences differ only in the edited value's own tokens."""
    a = encode(tokenizer, _answer_prefix(base['prompt'], tokenizer, True))
    b = encode(tokenizer, _answer_prefix(edit['prompt'], tokenizer, True))
    _, ma, mb = middle(a, b)
    return (tokenizer.decode(ma).strip() == base['source_value']
            and tokenizer.decode(mb).strip() == base['replacement_value'])


def name_value_distance(tokenizer, line, entity, value):
    """Tokens strictly between the last token of the entity name and the first token of the value."""
    ids = encode(tokenizer, line)
    name_end = next(k for k in range(len(ids) + 1) if entity in tokenizer.decode(ids[:k]))
    value_end = next(k for k in range(len(ids) + 1) if value in tokenizer.decode(ids[:k]))
    value_start = max(k for k in range(value_end) if value in tokenizer.decode(ids[k:value_end]))
    return value_start - name_end


def audit_candidate(template, tokenizers, histories):
    rows = generate_distance('pilot', histories, seed=20261106, superseded_near=template)
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r['history_id'], r['condition'], r['distance'], r['historical_order'], r['current_order'],
               r['edited_variable'], r['query_variable'])][r['edited']] = r
    result = {'template': template, 'tokenizers': {}}
    for name, tok in tokenizers.items():
        failures = sum(not span_ok(tok, m[0], m[1]) for m in pairs.values())
        distances = defaultdict(list)
        for r in rows:
            if r['edited'] or r['query_variable'] != 'x' or r['edited_variable'] != 'x':
                continue
            for line in r['prompt'].split('\n')[:2]:
                entity = next(e for e in r['entities'] if e in line)
                distances[f"{r['condition']}_{r['distance']}"].append(
                    name_value_distance(tok, line, entity, r['historical_values'][entity]))
        result['tokenizers'][name] = {'pairs_checked': len(pairs), 'span_failures': failures,
            'name_value_token_distance': {k: {'median': statistics.median(v), 'min': min(v), 'max': max(v)}
                                          for k, v in sorted(distances.items())}}
    result['passes_span_audit'] = all(t['span_failures'] == 0 for t in result['tokenizers'].values())
    # Amendment 1: the near template must also be measurably nearer than the far one in every tokenizer.
    result['shorter_than_far'] = all(
        t['name_value_token_distance']['superseded_near']['median']
        < t['name_value_token_distance']['superseded_far']['median'] for t in result['tokenizers'].values())
    result['passes'] = result['passes_span_audit'] and result['shorter_than_far']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--histories', type=int, default=96)
    a = p.parse_args()
    tokenizers = {Path(c).stem: load_tokenizer(c) for c in CONFIGS}
    candidates = [audit_candidate(t, tokenizers, a.histories) for t in SUPERSEDED_NEAR_CANDIDATES]
    chosen = next((c['template'] for c in candidates if c['passes']), None)
    report = {'rule': ('first candidate, in recorded order, with zero span failures for both tokenizers '
                       'and (amendment 1) a smaller median name-value token distance than the far '
                       'superseded template in both tokenizers'),
              'audit_dataset': 'distance pilot-stage histories (seed 20261106), not used for scoring',
              'candidates': candidates, 'chosen_superseded_near': chosen}
    write_new(a.output, report)
    print(json.dumps(report, indent=2))
    if chosen is None:
        raise SystemExit('no near superseded candidate passes the tokenizer audit')


if __name__ == '__main__':
    main()
