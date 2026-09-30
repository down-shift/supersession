#!/usr/bin/env python3
"""Resumable next-token scoring for explicit controls/status matched edit pairs."""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from tqdm.auto import tqdm
from src.data.io import read_jsonl, sha256_file
from src.data.progress import prepare_jsonl_progress, append_jsonl_record
from src.data.supersession_behavior import (SCHEMA, audit_behavior_dataset, audit_candidate_tokens,
                                          audit_tokenized_pairs, render_behavior_example)
from src.experiments.behavior import score_example
from src.analysis.supersession_behavior import matched_edit_effect
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json


def score_dataset(model, tokenizer, rows, token_ids, output, completed, chat=True):
    """Use the shared scorer and checkpoint format, ordering baseline before edit."""
    expected = {r['example_id']: r for r in rows}
    saved = read_jsonl(output) if Path(output).exists() else []
    baselines = {}
    for score in saved:
        eid = score['example_id']
        if eid not in completed or any(score.get(k) != v for k, v in expected[eid].items()):
            raise ValueError('checkpoint semantic metadata differs from dataset')
        if score.get('prompt') != render_behavior_example(expected[eid], tokenizer, chat):
            raise ValueError('checkpoint rendered prompt differs from current tokenizer/template')
        if score['pair_direction'] == 0:
            baselines[score['pair_id']] = score
    for score in saved:
        if score['pair_direction'] == 1 and score['pair_id'] not in baselines:
            raise ValueError('checkpoint edited member lacks its baseline')
    ordered = sorted(rows, key=lambda r: (r['pair_id'], r['pair_direction']))
    for row in tqdm(ordered, desc='Controls/status behavior scoring'):
        if row['example_id'] in completed:
            continue
        result = score_example(model, tokenizer, row, token_ids, chat=chat,
                               renderer=render_behavior_example, preserve_metadata=True)
        if row['pair_direction'] == 0:
            baselines[row['pair_id']] = result
        else:
            result['matched_edit_effect'] = matched_edit_effect(baselines[row['pair_id']], result)
        append_jsonl_record(output, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/four_query_288.yaml')
    p.add_argument('--dataset', required=True)
    p.add_argument('--token-ids', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    c = load_config(a.config)
    rows = read_jsonl(a.dataset)
    kind = audit_behavior_dataset(rows)
    token_doc = json.loads(Path(a.token_ids).read_text())
    ids = token_doc['token_ids']
    audit_candidate_tokens(rows, ids)  # Fail on missing source/replacement/answer before model load.
    if not a.resume and any(Path(str(a.output)+suffix).exists() for suffix in ('', '.run.json', '.provenance.json')):
        raise FileExistsError(f'{a.output} already exists; choose a new output or pass --resume')
    if a.resume and not Path(a.output+'.run.json').exists():
        raise ValueError('--resume requires the original .run.json manifest')
    model, tok = load_model(c)
    for key in ('model_revision', 'tokenizer_revision'):
        # A requested immutable commit also identifies a tokenizer whose loader
        # does not expose _commit_hash. Never label a mutable branch as resolved.
        requested = c['model'].get(key) or c['model'].get('revision')
        if not c.get('resolved_'+key) and requested and re.fullmatch(r'[0-9a-f]{40}', requested):
            c['resolved_'+key] = requested
        if not re.fullmatch(r'[0-9a-f]{40}', str(c.get('resolved_'+key))):
            raise ValueError(f'cannot record exact {key}; pin an immutable revision')
    for key in ('model_revision', 'tokenizer_revision'):
        frozen = token_doc.get(key)
        if frozen and c.get('resolved_'+key) != frozen:
            raise ValueError(f'validated token-map {key} does not match loaded model/tokenizer')
    chat = c['model'].get('chat_template', True)
    alignment = audit_tokenized_pairs(rows, tok, ids, chat)
    run = provenance(c, a.dataset)
    code_paths = [Path(__file__), Path('src/data/supersession_behavior.py'), Path('src/experiments/behavior.py'),
                  Path('src/data/progress.py'), Path('src/data/token_validation.py'), Path('src/analysis/supersession_behavior.py'),
                  Path('src/analysis/metrics.py'), Path('src/models/loader.py')]
    code_sha = hashlib.sha256(b''.join(path.read_bytes() for path in code_paths)).hexdigest()
    fingerprint_config = copy.deepcopy(c)
    fingerprint_config['supersession_scoring'] = {'schema': SCHEMA, 'kind': kind,
        'config_sha256': sha256_file(a.config), 'code_sha256': code_sha,
        'chat_template_sha256': hashlib.sha256(str(getattr(tok, 'chat_template', None)).encode()).hexdigest(),
        'python': run['python'], 'packages': run['packages']}
    completed = prepare_jsonl_progress(a.output, a.dataset, a.token_ids, fingerprint_config, rows, resume=a.resume)
    sidecar = Path(a.output+'.provenance.json')
    if a.resume and completed and not sidecar.exists():
        raise ValueError('checkpoint provenance is missing; preserve outputs and use a fresh path')
    if not sidecar.exists():
        run.update(experiment_kind=kind, schema=SCHEMA, candidate_token_ids=ids,
                   dataset_seed=sorted({r['seed'] for r in rows}), config_sha256=sha256_file(a.config),
                   token_map_sha256=sha256_file(a.token_ids), code_sha256=code_sha,
                   prompt_alignment_audit=alignment, primary_filter='all_valid_trials')
        save_json(run, sidecar)
    score_dataset(model, tok, rows, ids, a.output, completed, chat)
    print(f'scored/resumed {len(rows)} {kind} members; {len(completed)} loaded from checkpoint')


if __name__ == '__main__':
    main()
