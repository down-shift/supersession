"""stale_decision_v2f preflight (paper_v2/sections/A-protocol-history.tex), run before any development scoring.

    python -m scripts.stale_decision_v2f_preflight causality --config CFG --dataset DIR/development.jsonl --output DIR/causality_check.json
    python -m scripts.stale_decision_v2f_preflight format --config CFG --smoke DIR/smoke --output DIR/format_check.json

causality: (a) load path and model class, (b) the rendered chat template with thinking disabled, (d) for 12
development records and two queries each, the maximum absolute change in the top-20 next-token log
probabilities at the last prompt position when 1 or 2 tokens are appended. The amendment requires exactly 0.0.
format: (c) for the 24-record smoke run, the exact generated token ids and decoded text after each label, and
whether every generation ends with the configured end-of-turn token; (f) the largest canonical valid mass of the
two terminated action events and the largest summed mass of the four retrieval candidates (both must be at most 1
up to the protocol's rounding tolerance of 1e-5).
"""
import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

from src.data.stale_decision_v2 import build_prompt, read_rows, write_new
from src.experiments import stale_decision_v2f as exp
from src.experiments.stale_decision_v1 import rendered
from src.cross_model.tokens import encode

TOLERANCE = 1e-5


def causality(config, dataset, output):
    import torch
    model, tok = exp.load_model(copy.deepcopy(config))
    rows = read_rows(dataset)
    device = model.get_input_embeddings().weight.device
    checks = []
    for r in rows[:12]:
        for q in ('easy', 'downstream'):
            p = rendered(tok, build_prompt(r, q)[0])
            ids = encode(tok, p)
            ext = encode(tok, p + ('S1' if r['vocabulary'] == 'opaque' else r['actions'][0]))[len(ids):]
            with torch.inference_mode():
                lp = [model(input_ids=torch.tensor([ids + ext[:k]], device=device), use_cache=False)
                      .logits[0, len(ids) - 1].float().log_softmax(-1) for k in (0, 1, 2)]
            top = lp[0].topk(20).indices
            checks.append({'example_id': r['example_id'], 'query': q, 'appended': len(ext[:2]),
                           'max_abs_top20': max(float((lp[0][top] - x[top]).abs().max()) for x in lp[1:])})
    worst = max(c['max_abs_top20'] for c in checks)
    sample = rendered(tok, build_prompt(rows[0], 'downstream')[0])
    write_new(output, {'checks': checks, 'max_abs_top20': worst, 'exactly_zero': worst == 0.0, 'passed': worst == 0.0,
                       'model_class': type(model).__name__, 'load_class': config['model']['load_class'],
                       'tokenizer_kwargs': config['model']['tokenizer_kwargs'], 'template_tail': sample[-300:],
                       'code_hash': exp.code_hash(), 'runtime': config['model']})
    print('causality max', worst, type(model).__name__)


def format_check(config, smoke, output):
    rs = [json.loads(l) for l in open(Path(smoke) / 'scores.jsonl')]
    easy = [json.loads(l) for l in open(Path(smoke) / 'scores.jsonl.easy.jsonl')]
    end_id = rs[0]['eos_token_id']
    endings = Counter((r['raw_text'], tuple(r['generated_token_ids'][-3:])) for r in rs)
    terminated = all(r['terminated'] and r['generated_token_ids'][-1] == end_id for r in rs)
    easy_terminated = all(e['easy_terminated'] for e in easy)
    max_valid = max(r['canonical_valid_mass'] for r in rs)
    min_valid = min(r['canonical_valid_mass'] for r in rs)
    max_easy = max(sum(math.exp(v) for v in e[f].values()) for e in easy for f in ('easy_target', 'easy_other'))
    ok = terminated and easy_terminated and min_valid > 0.5 and max_valid <= 1 + TOLERANCE and max_easy <= 1 + TOLERANCE
    write_new(output, {'end_token_id': end_id, 'end_token': config['model']['end_token'],
                       'end_suffix': config['model']['end_suffix'], 'terminated_with_end_token': terminated,
                       'easy_terminated': easy_terminated, 'min_canonical_valid_mass': min_valid,
                       'max_canonical_valid_mass': max_valid, 'max_four_candidate_mass': max_easy,
                       'generated_endings': [{'raw_text': k[0], 'last_ids': list(k[1]), 'count': n}
                                             for k, n in endings.most_common()],
                       'passed': ok, 'code_hash': exp.code_hash()})
    print('FORMAT', 'passed' if ok else 'FAILED', 'terminated', terminated, 'valid mass', round(min_valid, 4),
          round(max_valid, 6), 'four-candidate max', round(max_easy, 6))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['causality', 'format'])
    p.add_argument('--config', required=True)
    p.add_argument('--dataset')
    p.add_argument('--smoke')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    exp.check_config(config)
    if a.command == 'causality':
        causality(config, a.dataset, a.output)
    elif not format_check(config, a.smoke, a.output):
        raise SystemExit(3)


if __name__ == '__main__':
    main()
