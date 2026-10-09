"""stale_decision_v2 CLI (docs/stale_decision_v2.md); same commands as the v1b CLI.

    python -m scripts.stale_decision_v2 generate --output DIR/development.jsonl
    python -m scripts.stale_decision_v2 token-audit --dataset DIR/development.jsonl --output DIR/token_audit.json
    python -m scripts.stale_decision_v2 freeze --output DIR/freeze.json
    python -m scripts.stale_decision_v2 verify-scorer --dataset ... --output DIR/scorer_check.json
    python -m scripts.stale_decision_v2 benchmark --dataset ... --freeze ... --limit 40 --output DIR/bench/scores.jsonl
    python -m scripts.stale_decision_v2 score --dataset ... --freeze ... --output DIR/development_scores.jsonl [--resume]
    python -m scripts.stale_decision_v2 analyze --dataset ... --scores ... --output DIR/development_report.json
"""
import argparse
import copy
import json
import time
from pathlib import Path

from src.data.stale_decision_v2 import GRID, digest, generate, read_rows, validate, verify_no_overlap, write_new
from src.experiments import stale_decision_v2 as exp

COUNT = {'development': 'development_replicates', 'frozen_gate': 'gate_replicates', 'confirmation': 'confirmation_replicates'}


def check_dataset(path, config):
    rows = read_rows(path)
    split = rows[0]['split']
    validate(rows, expected_histories=len(GRID) * config[COUNT[split]])
    if digest(rows) != digest(generate(split, config[COUNT[split]])):
        raise ValueError('dataset differs from deterministic generation')
    manifest = json.loads(Path(path + '.manifest.json').read_text())
    if (manifest['dataset_hash'] != digest(rows) or manifest['config_hash'] != digest(config) or
            manifest['code_hash'] != exp.code_hash()):
        raise ValueError('dataset provenance differs from the active v1b protocol')
    return rows


def check_freeze(path, config):
    frozen = json.loads(Path(path).read_text())
    if frozen['code_hash'] != exp.code_hash() or frozen['config_hash'] != digest(config):
        raise ValueError('freeze differs from the active v1b protocol')


def load(config):
    from src.cross_model.runtime import load_pinned_model
    return load_pinned_model(copy.deepcopy(config))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['generate', 'token-audit', 'freeze', 'verify-scorer', 'benchmark', 'score', 'analyze'])
    p.add_argument('--config', default='configs/stale_decision_v2/protocol.json')
    p.add_argument('--split', choices=list(COUNT), default='development')
    p.add_argument('--dataset')
    p.add_argument('--scores')
    p.add_argument('--freeze')
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--limit', type=int)
    p.add_argument('--scorer-check')
    p.add_argument('--gate-dataset')
    p.add_argument('--gate-scores')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    exp.check_config(config)
    if a.command == 'freeze':
        write_new(a.output, {'protocol': exp.PROTOCOL, 'config_hash': digest(config), 'code_hash': exp.code_hash(),
                             'status': 'frozen', 'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        return
    if a.command == 'generate':
        if a.split != 'development' and not a.exclude:
            p.error('gate and confirmation splits require --exclude of every prior split')
        if a.split == 'confirmation':
            # Recompute the gate from its saved dataset and completed scores; never trust a pass flag.
            if not (a.gate_dataset and a.gate_scores):
                p.error('confirmation requires --gate-dataset and --gate-scores')
            from src.analysis import stale_decision_v1 as v1a
            from src.analysis.stale_decision_v2 import v1b_validation
            gate_rows = check_dataset(a.gate_dataset, config)
            if {r['split'] for r in gate_rows} != {'frozen_gate'}:
                raise ValueError('gate dataset is not the frozen gate split')
            _, gate_scores = exp.verify_complete_run(a.gate_scores, gate_rows, config)
            with v1b_validation():
                gate = v1a.gates(gate_rows, gate_scores)
            if not gate['any_score_pass']:
                raise ValueError('no passing recomputed gate; confirmation not generated')
            if not any(digest(read_rows(e)) == digest(gate_rows) for e in a.exclude):
                raise ValueError('gate histories must be excluded')
        rows = generate(a.split, config[COUNT[a.split]])
        prior = [check_dataset(e, config) for e in a.exclude]
        verify_no_overlap(*prior, rows)
        write_new(a.output + '.manifest.json', {'dataset_hash': digest(rows), 'config_hash': digest(config),
                                                'code_hash': exp.code_hash(), 'split': a.split, 'seed': rows[0]['seed'],
                                                'excluded_hashes': [digest(r) for r in prior]})
        with Path(a.output).open('x') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps({'histories': len({r['history_id'] for r in rows}), 'records': len(rows)}))
        return
    rows = check_dataset(a.dataset, config)
    if a.command == 'token-audit':
        from transformers import AutoTokenizer
        m = config['model']
        tok = AutoTokenizer.from_pretrained(m['tokenizer_id'], revision=m['tokenizer_revision'])
        write_new(a.output, exp.token_audit(rows, tok))
    elif a.command == 'verify-scorer':
        model, tok = load(config)
        fast = exp.score_batched if config['scorer'] == 'batched' else exp.score_cached
        write_new(a.output, {**exp.verify_cached_scorer(model, tok, rows, fast_scorer=fast), 'scorer': config['scorer'],
                             'code_hash': exp.code_hash(),
                             'config_hash': digest(config)})
    elif a.command in ('benchmark', 'score'):
        check_freeze(a.freeze, config)
        if config['scorer'] in ('kv_cache', 'batched'):
            if not a.scorer_check:
                p.error('a fast scorer requires --scorer-check from verify-scorer')
            check = json.loads(Path(a.scorer_check).read_text())
            if (not check['passed'] or check.get('scorer') != config['scorer'] or check['code_hash'] != exp.code_hash() or
                    check['config_hash'] != digest(config)):
                raise ValueError('no passing scorer check for the active code and config')
        model, tok = load(config)
        start = time.perf_counter()
        exp.run(rows, config, a.output, model, tok, a.resume, limit=a.limit if a.command == 'benchmark' else None)
        elapsed = time.perf_counter() - start
        n = a.limit if a.command == 'benchmark' else len(rows)
        write_new(a.output + '.timing.json', {'records': n, 'seconds': elapsed, 'seconds_per_record': elapsed / n,
                                              'note': 'includes the tokenizer audit of the full dataset'})
    elif a.command == 'analyze':
        from src.analysis.stale_decision_v2 import analyze
        _, scores = exp.verify_complete_run(a.scores, rows, config)
        result = analyze(rows, scores)
        result.update(dataset_hash=digest(rows), scores_hash=digest(scores), code_hash=exp.code_hash(),
                      config_hash=digest(config))
        write_new(a.output, result)


if __name__ == '__main__':
    main()
