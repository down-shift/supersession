"""stale_decision_v2s CLI (paper_v2/sections/A-protocol-history.tex, part B): static other-attribute construction on the
v2 confirmation routing histories.

    python -m scripts.stale_decision_v2s generate --config CFG --output DIR/static.jsonl
    python -m scripts.stale_decision_v2s freeze --config CFG --output DIR/freeze.json
    python -m scripts.stale_decision_v2s verify-scorer --config CFG --dataset DIR/static.jsonl --output DIR/scorer_check.json
    python -m scripts.stale_decision_v2s score --config CFG --dataset ... --freeze ... --scorer-check ... --output DIR/static_scores.jsonl [--resume]
    python -m scripts.stale_decision_v2s analyze --config CFG --dataset ... --scores ... --sealed-dataset CONF/confirmation.jsonl \
        --sealed-scores CONF/confirmation_scores.jsonl --output DIR/static_report.json

The preflight causality check of the families amendment is run with the v2f preflight script's method
(`causality` below), on the static-other rows.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from src.data.stale_decision_v2s import digest, generate, read_rows, validate, write_new
from src.experiments import stale_decision_v2s as exp


def check_dataset(path, config):
    rows = read_rows(path)
    validate(rows)
    if digest(rows) != digest(generate()):
        raise ValueError('dataset differs from deterministic generation')
    manifest = json.loads(Path(path + '.manifest.json').read_text())
    if manifest['dataset_hash'] != digest(rows) or manifest['code_hash'] != exp.code_hash():
        raise ValueError('dataset provenance differs from the active protocol')
    return rows


def load(config):
    model, tok = exp.load_model(copy.deepcopy(config))
    return model, exp.EndOfTurn(tok, config['model']['end_token'], config['model']['end_suffix'])


def causality(model, tok, rows):
    """Top-20 next-token log probabilities at the last prompt position, with 1 or 2 tokens appended (must be 0.0)."""
    import torch
    from src.cross_model.tokens import encode
    from src.experiments.stale_decision_v1 import rendered
    from src.data.stale_decision_v2s import build_prompt
    device = model.get_input_embeddings().weight.device
    worst = 0.0
    for r in [x for x in rows if x['family'] == 'static_other'][:12]:
        for q in ('easy', 'downstream'):
            p = rendered(tok, build_prompt(r, q)[0])
            ids = encode(tok, p)
            ext = encode(tok, p + ('S1' if r['vocabulary'] == 'opaque' else r['actions'][0]))[len(ids):]
            with torch.inference_mode():
                lp = [model(input_ids=torch.tensor([ids + ext[:k]], device=device), use_cache=False)
                      .logits[0, len(ids) - 1].float().log_softmax(-1) for k in (0, 1, 2)]
            top = lp[0].topk(20).indices
            worst = max(worst, max(float((lp[0][top] - x[top]).abs().max()) for x in lp[1:]))
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['generate', 'freeze', 'verify-scorer', 'score', 'analyze'])
    p.add_argument('--config', required=True)
    p.add_argument('--dataset')
    p.add_argument('--scores')
    p.add_argument('--freeze')
    p.add_argument('--scorer-check')
    p.add_argument('--sealed-dataset')
    p.add_argument('--sealed-scores')
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
        rows = generate()
        write_new(a.output + '.manifest.json', {'dataset_hash': digest(rows), 'code_hash': exp.code_hash(),
                                                'records': len(rows)})
        with Path(a.output).open('x') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps({'records': len(rows)}))
        return
    rows = check_dataset(a.dataset, config)
    if a.command == 'verify-scorer':
        model, tok = load(config)
        fast = exp.score_batched if config['scorer'] == 'batched' else exp.score_cached
        static = [r for r in rows if r['family'] == 'static_other']
        check = exp.verify_cached_scorer(model, tok, static, fast_scorer=fast)
        worst = causality(model, tok, rows)
        check.update(scorer=config['scorer'], code_hash=exp.code_hash(), config_hash=digest(config),
                     causality_max_abs_top20=worst, causality_exactly_zero=worst == 0.0)
        check['passed'] = check['passed'] and worst == 0.0
        write_new(a.output, check)
        print('scorer', check['max_abs_diff'], 'causality', worst, 'passed', check['passed'])
    elif a.command == 'score':
        frozen = json.loads(Path(a.freeze).read_text())
        if frozen['code_hash'] != exp.code_hash() or frozen['config_hash'] != digest(config):
            raise ValueError('freeze differs from the active protocol')
        check = json.loads(Path(a.scorer_check).read_text())
        if (not check['passed'] or check.get('scorer') != config['scorer'] or check['code_hash'] != exp.code_hash() or
                check['config_hash'] != digest(config)):
            raise ValueError('no passing scorer and causality check for the active code and config')
        model, tok = load(config)
        start = time.perf_counter()
        exp.run(rows, config, a.output, model, tok, a.resume)
        elapsed = time.perf_counter() - start
        write_new(a.output + '.timing.json', {'records': len(rows), 'seconds': elapsed, 'seconds_per_record': elapsed / len(rows),
                                              'note': 'includes the tokenizer audit of the full dataset'})
    elif a.command == 'analyze':
        _, scores = exp.verify_complete_run(a.scores, rows, config)
        complete = json.loads(Path(a.sealed_scores + '.complete.json').read_text())
        if hashlib.sha256(Path(a.sealed_scores).read_bytes()).hexdigest() != complete['scores_sha256']:
            raise ValueError('sealed scores differ from their completion record')
        sealed_rows, sealed_scores = read_rows(a.sealed_dataset), read_rows(a.sealed_scores)
        if {s['model_id'] for s in sealed_scores} != {config['model']['id']}:
            raise ValueError('sealed scores come from a different model')
        result = exp.analyze_static(rows, scores, sealed_rows, sealed_scores)
        result.update(dataset_hash=digest(rows), scores_hash=digest(scores), code_hash=exp.code_hash(),
                      config_hash=digest(config), sealed_scores_sha256=complete['scores_sha256'])
        write_new(a.output, result)


if __name__ == '__main__':
    main()
