#!/usr/bin/env python3
"""Competence-only template development or strict frozen gate; never reads R metrics."""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.data.io import read_jsonl
from src.utils import save_json

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True)
p.add_argument('--behavior', required=True)
p.add_argument('--stage', choices=('development', 'frozen_gate'), required=True)
p.add_argument('--output', required=True)
p.add_argument('--threshold', type=float, default=.99)
a = p.parse_args()
if a.threshold != .99:
    raise ValueError('preregistered competence gate is fixed at .99')
ds, scores = read_jsonl(a.dataset), read_jsonl(a.behavior)
if len(ds) != len(scores) or any(x.get('example_id') != y.get('example_id') for x,y in zip(ds,scores)):
    raise ValueError('scored records do not exactly match the dataset ordering')
groups = defaultdict(list)
for x, y in zip(ds, scores):
    if any(y.get(k) != x.get(k) for k in ('example_id','condition','history_id','prompt_variant','answer')):
        raise ValueError('scored metadata mismatch')
    groups[(x['prompt_variant'], x['condition'])].append(y)
summary = {f'{t}|{c}': {'n':len(rows), 'full_vocab_accuracy':float(np.mean([r['full_vocab_next_token_accuracy'] for r in rows])),
                         'candidate_accuracy':float(np.mean([r['accuracy'] for r in rows])),
                         'mean_candidate_rank':float(np.mean([r['candidate_rank'] for r in rows])),
                         'mean_current_minus_stale_logit_margin': (float(np.mean([r['candidate_logits'][r['answer']] - r['candidate_logits'][r['stale_value']] for r in rows if r.get('stale_value')])) if any(r.get('stale_value') for r in rows) else None)}
           for (t,c), rows in sorted(groups.items())}
passed = bool(summary) and all(v['full_vocab_accuracy'] >= a.threshold and
                                v['candidate_accuracy'] >= a.threshold and v['mean_candidate_rank'] <= 1.01 and
                                (v['mean_current_minus_stale_logit_margin'] is None or v['mean_current_minus_stale_logit_margin'] > 0)
                                for v in summary.values())
save_json({'stage':'frozen_competence_gate' if a.stage == 'frozen_gate' else 'development_competence_only',
           'pass': passed if a.stage == 'frozen_gate' else None,
           'preregistered_gate': {'full_vocab_accuracy_min':.99, 'candidate_accuracy_min':.99,
                                  'mean_candidate_rank_max':1.01, 'stale_margin_min':0.0},
           'causal_metrics_inspected':False, 'n_records':len(scores), 'by_template_condition':summary}, a.output)
print(json.dumps({'stage':a.stage,'pass':passed if a.stage == 'frozen_gate' else None,'cells':len(summary)}, sort_keys=True))
