"""Isolated protocol CLI; no inference during generation/validation/tests."""
import argparse
import json
from pathlib import Path
from src.data.stale_decision_v1 import GRID, generate, validate, verify_no_overlap, read_rows, write_new, digest
from src.experiments.stale_decision_v1 import check_config, code_hash, token_audit, verify_complete_run


def expected_histories(split, config):
    key = {'development': 'development_replicates', 'frozen_gate': 'gate_replicates',
           'confirmation': 'confirmation_replicates'}[split]
    return len(GRID) * config[key]


def validate_frozen_dataset(rows, config):
    validate(rows)
    split = rows[0]['split']
    count_key = {'development': 'development_replicates', 'frozen_gate': 'gate_replicates',
                 'confirmation': 'confirmation_replicates'}[split]
    validate(rows, expected_histories=expected_histories(split, config))
    if digest(rows) != digest(generate(split, config[count_key])):
        raise ValueError('dataset differs from deterministic frozen generation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['generate', 'validate', 'token-audit', 'score', 'analyze', 'plan', 'freeze'])
    p.add_argument('--config', default='configs/stale_decision_v1/protocol.json')
    p.add_argument('--split', choices=['development', 'frozen_gate', 'confirmation'], default='development')
    p.add_argument('--dataset'); p.add_argument('--scores'); p.add_argument('--output', required=True)
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--freeze'); p.add_argument('--gate'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); config = json.loads(Path(a.config).read_text()); check_config(config)
    if a.resume and a.command != 'score':
        p.error('--resume only applies to score')
    if a.command == 'freeze':
        write_new(a.output, {'protocol': config['protocol'], 'config_hash': digest(config), 'code_hash': code_hash(),
                            'confirmation_rules_recorded': True, 'status': 'frozen; inference not executed'})
        return
    if a.command == 'generate':
        if a.split != 'development':
            if not a.freeze or not a.exclude:
                p.error('gate/confirmation requires --freeze and --exclude prior datasets')
            freeze = json.loads(Path(a.freeze).read_text())
            if freeze['code_hash'] != code_hash() or freeze['config_hash'] != digest(config):
                raise ValueError('freeze differs from active protocol')
        if a.split == 'confirmation':
            if not a.gate:
                p.error('confirmation requires --gate report')
            gate = json.loads(Path(a.gate).read_text())
            # Recompute eligibility from referenced raw gate data/scores; never trust a pass flag.
            from src.analysis.stale_decision_v1 import gates
            gr = read_rows(gate['dataset_path']); gs = read_rows(gate['scores_path'])
            validate_frozen_dataset(gr, config)
            gate_manifest = json.loads(Path(gate['dataset_path']+'.manifest.json').read_text())
            if (gate_manifest.get('dataset_hash') != digest(gr) or gate_manifest.get('config_hash') != digest(config) or
                    gate_manifest.get('code_hash') != code_hash() or {r['split'] for r in gr} != {'frozen_gate'} or
                    gate['config_hash'] != digest(config) or gate['code_hash'] != code_hash()):
                raise ValueError('invalid frozen gate lineage')
            if digest(gr) != gate['dataset_hash'] or digest(gs) != gate['scores_hash'] or not gates(gr, gs)['any_score_pass']:
                raise ValueError('no passing recomputed gate')
            verify_complete_run(gate['scores_path'], gr, config)
        count_key = {'development': 'development_replicates', 'frozen_gate': 'gate_replicates', 'confirmation': 'confirmation_replicates'}[a.split]
        rows = generate(a.split, config[count_key]); prior = [read_rows(e) for e in a.exclude]
        for exclude_path, ds in zip(a.exclude, prior):
            validate_frozen_dataset(ds, config)
            prior_manifest = json.loads(Path(exclude_path+'.manifest.json').read_text())
            if (prior_manifest.get('dataset_hash') != digest(ds) or
                    prior_manifest.get('config_hash') != digest(config) or
                    prior_manifest.get('code_hash') != code_hash()):
                raise ValueError('excluded dataset provenance differs')
        prior_splits = {r['split'] for ds in prior for r in ds}
        required = {'development', 'frozen_gate'} if a.split == 'confirmation' else {'development'} if a.split == 'frozen_gate' else set()
        if not required <= prior_splits:
            raise ValueError('all prior splits must be excluded')
        if a.split == 'confirmation' and not any(digest(ds) == gate['dataset_hash'] for ds in prior):
            raise ValueError('passing gate histories missing from exclusions')
        verify_no_overlap(*prior, rows)
        path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or Path(str(path)+'.manifest.json').exists():
            raise FileExistsError(path)
        write_new(str(path)+'.manifest.json', {'dataset_hash': digest(rows), 'config_hash': digest(config),
                  'code_hash': code_hash(), 'seed': rows[0]['seed'], 'split': a.split,
                  'excluded_hashes': [digest(r) for r in prior]})
        with path.open('x') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps({'histories': len({r['history_id'] for r in rows}), 'records': len(rows), 'no_inference': True}))
        return
    if not a.dataset:
        p.error('--dataset required')
    rows = read_rows(a.dataset); validate_frozen_dataset(rows, config)
    sidecar = json.loads(Path(a.dataset+'.manifest.json').read_text())
    if sidecar['dataset_hash'] != digest(rows) or sidecar['config_hash'] != digest(config) or sidecar['code_hash'] != code_hash():
        raise ValueError('dataset provenance differs')
    if a.command == 'validate':
        verify_no_overlap(*[read_rows(e) for e in a.exclude], rows)
        write_new(a.output, {'valid': True, 'records': len(rows), 'dataset_hash': digest(rows), 'no_inference': True})
    elif a.command == 'plan':
        # Upper bound for full teacher-forced multi-token candidate events, plus 16 generation steps each.
        n = len(rows)
        write_new(a.output, {'records': n, 'histories': len({r['history_id'] for r in rows}),
                  'scoring_forward_upper_bound': n * (3 + 2 + 2*16),
                  'generation_forward_upper_bound': n * 32, 'total_forward_upper_bound': n * 69,
                  'formula': '3 initial + 2 action suffix + 2*16 easy surfaces; two generations <=16 steps',
                  'json_disk_estimate_bytes': n * 12000, 'no_inference': True})
    elif a.command in ('token-audit', 'score'):
        if a.command == 'token-audit':
            from transformers import AutoTokenizer
            m = config['model']
            tok = AutoTokenizer.from_pretrained(m.get('tokenizer_id', m['id']), revision=m['tokenizer_revision'], local_files_only=True)
            write_new(a.output, token_audit(rows, tok))
        else:
            if not a.freeze:
                p.error('score requires --freeze')
            frozen = json.loads(Path(a.freeze).read_text())
            if frozen['code_hash'] != code_hash() or frozen['config_hash'] != digest(config):
                raise ValueError('score freeze mismatch')
            from src.cross_model.runtime import load_pinned_model
            from src.experiments.stale_decision_v1 import run
            import copy
            model, tok = load_pinned_model(copy.deepcopy(config))
            run(rows, config, a.output, model, tok, a.resume)
    elif a.command == 'analyze':
        if not a.scores:
            p.error('--scores required')
        from src.analysis.stale_decision_v1 import analyze
        _, scores = verify_complete_run(a.scores, rows, config)
        result = analyze(rows, scores)
        result.update(dataset_path=str(Path(a.dataset).resolve()), scores_path=str(Path(a.scores).resolve()),
                      dataset_hash=digest(rows), scores_hash=digest(scores), code_hash=code_hash(), config_hash=digest(config))
        write_new(a.output, result)


if __name__ == '__main__':
    main()
