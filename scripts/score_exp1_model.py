#!/usr/bin/env python3
"""Score an additional model on Experiment 1's sealed screening and confirmation datasets (plan P2).

The rows, prompt renderer (`robustness_v2.render`), candidate continuations, scorer (`score_row`),
competence screen (`robustness_protocol.evaluate`: 99% strict rank-one per construction on distinct
prompts, positive superseded margin) and confirmation analysis (`robustness_analysis.report`) are
the ones used for Qwen3-8B and Gemma 3 4B. Only the model differs. The original sealed artifacts
are read, never written.

    # 1. competence screen on the byte-identical screening dataset
    python scripts/score_exp1_model.py --config C.yaml --stage gate --dataset GATE.jsonl --out DIR
    # 2. confirmation, only after a passing screen (or --descriptive for a model that failed it)
    python scripts/score_exp1_model.py --config C.yaml --stage confirmatory --dataset CONF.jsonl --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

from src.cross_model import robustness_protocol as design
from src.cross_model import robustness_v2 as data
from src.cross_model.protocol import digest, read_sealed, sealed, write_new
from src.cross_model.score_checks import checked_scores
from src.cross_model.tokens import validate as validate_tokens
from src.data.io import read_jsonl, sha256_file
from src.utils import load_config

CODE = ('scripts/score_exp1_model.py', 'src/cross_model/robustness_v2.py', 'src/cross_model/robustness_protocol.py',
        'src/cross_model/robustness_analysis.py', 'src/cross_model/scoring.py', 'src/cross_model/tokens.py')
EXPECTED_SHA256 = {'frozen_gate': 'a28809c59d377221', 'confirmatory': '27c1e9b80e62a0cb'}  # sealed dataset prefixes


def score_all(model, tok, rows, candidate, competence_only, path, generate=False):
    from src.cross_model.scoring import score_row
    done = {r['example_id']: r for r in read_jsonl(path)} if path.exists() else {}
    cache = {r['prompt']: r for r in done.values()}
    with path.open('a') as handle:
        for i, row in enumerate(rows):
            if row['example_id'] in done:
                continue
            prompt = data.render(row, tok, True)
            if prompt not in cache:
                cache[prompt] = score_row(model, tok, row, candidate, competence_only, renderer=data.render)
                if generate:
                    from scripts.run_followups import _generate
                    from src.cross_model.followups import complete_answer_outcome
                    answer, n_tokens = _generate(model, tok, prompt, 32)
                    outcome = complete_answer_outcome(answer, row['answer'], None, row['candidate_values'])
                    cache[prompt].update(generated_answer=answer, parsed_answer=outcome['parsed_answer'],
                                         generation_token_count=n_tokens)
            record = {**cache[prompt], **row}
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            done[row['example_id']] = record
            if i % 500 == 0:
                print(f'{time.strftime("%H:%M:%S")} {i}/{len(rows)} rows, {len(cache)} distinct prompts', flush=True)
    return [done[r['example_id']] for r in rows]


def generation_summary(rows, scores):
    """Generated-answer outcomes per construction (docs/olmo_generation_v1.md), bootstrapped by history."""
    from collections import defaultdict

    from src.cross_model.followups import _bootstrap
    by_id = {s['example_id']: s for s in scores}
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ans = by_id[r['example_id']]['parsed_answer']
        key = (r['history_id'], r['condition'])
        matched = r['edited_variable'] == r['query']
        per[key]['correct_baseline' if r['pair_direction'] == 0 else 'correct_edited'].append(ans == r['answer'])
        if r['pair_direction'] == 0:
            per[key]['source_matched' if matched else 'source_other'].append(ans == r['source_value'])
        else:
            per[key]['donor_matched' if matched else 'donor_other'].append(ans == r['replacement_value'])
    out = {}
    for condition in sorted({c for _, c in per}):
        hist = []
        for (hid, c), metrics in per.items():
            if c != condition:
                continue
            h = {k: float(sum(v) / len(v)) for k, v in metrics.items()}
            if 'donor_matched' in h and 'donor_other' in h:
                h['donor_following'] = h['donor_matched'] - h['donor_other']
            if 'source_matched' in h and 'source_other' in h:
                h['source_following'] = h['source_matched'] - h['source_other']
            hist.append(h)
        out[condition] = {k: _bootstrap([h[k] for h in hist]) for k in hist[0]}
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        ans = by_id[r['example_id']]['parsed_answer']
        counts[r['condition']][0] += ans == r['answer']
        counts[r['condition']][1] += 1
    return {'by_condition': out, 'accuracy_counts': {c: {'correct': v[0], 'total': v[1]} for c, v in counts.items()},
            'definitions': {'source_matched': 'baseline answer = the edited entity\'s earlier value, edited entity queried',
                            'donor_following': 'P(answer = donor | edited, edited entity queried) - P(answer = donor | edited, other entity queried)'}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=('gate', 'confirmatory'), required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--descriptive', action='store_true', help='confirmation for a model that failed the screen')
    p.add_argument('--generate', action='store_true',
                   help='also record greedy generated answers (exploratory generation study, docs/olmo_generation_v1.md)')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    config = load_config(a.config)
    rows = read_jsonl(a.dataset)
    stage = rows[0]['cross_model_stage']
    if {'gate': 'frozen_gate'}.get(a.stage, a.stage) != stage:
        raise ValueError('dataset stage does not match --stage')
    if not sha256_file(a.dataset).startswith(EXPECTED_SHA256[stage]):
        raise ValueError('dataset is not the sealed Experiment 1 dataset for this stage')
    gate_path = out / 'gate_report.json'
    if a.stage == 'confirmatory':
        if not gate_path.exists():
            raise ValueError('confirmation requires this model\'s screening report')
        if not read_sealed(gate_path)['evaluation']['pass'] and not a.descriptive:
            raise ValueError('model failed the competence screen; confirmation only with --descriptive')
    import torch
    from src.cross_model.runtime import load_pinned_model
    t0 = time.perf_counter()
    model, tok = load_pinned_model(config)
    audit = validate_tokens(tok, rows, renderer=data.render, positions=data.semantic_positions)
    candidate = {'events': audit['events'], 'canonical_one_token_ids': audit['canonical_one_token_ids']}
    scores_path = out / f'{a.stage}{"_generation" if a.generate else ""}_scores.jsonl'
    scores = score_all(model, tok, rows, candidate, a.stage == 'gate', scores_path, a.generate)
    checked_scores(rows, scores, require_surfaces=True)
    prov = {'model': config['model'], 'dataset': str(Path(a.dataset).resolve()), 'dataset_sha256': sha256_file(a.dataset),
            'scores_sha256': sha256_file(scores_path), 'code_sha256': digest({c: sha256_file(c) for c in CODE}),
            'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'elapsed_seconds': time.perf_counter() - t0,
            'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'tokenizer_chat_template_sha256': digest(tok.chat_template)}
    if a.stage == 'gate':
        write_new(gate_path, sealed({'stage': 'frozen_gate', 'evaluation': design.evaluate(rows, scores, gate=True),
                                     'provenance': prov}))
        print(json.dumps(read_sealed(gate_path)['evaluation']['condition_summary'], indent=1))
        return
    if a.generate:
        write_new(out / 'generation_analysis.json', sealed({'provenance': prov, **generation_summary(rows, scores)}))
        print('generation study complete', out)
        return
    from src.cross_model.robustness_analysis import order_contrasts, report
    result = report(rows, scores)
    result.update(provenance=prov, descriptive_only=a.descriptive,
                  order_specific_contrasts=order_contrasts(result['history_rows']))
    write_new(out / 'confirmatory_analysis.json', sealed(result))
    with (out / 'per_history.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(result['history_rows'][0]))
        w.writeheader(); w.writerows(result['history_rows'])
    print('confirmation complete', out)


if __name__ == '__main__':
    main()
