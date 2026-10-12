"""stale_decision_think CLI (docs/stale_decision_think.md).

    python -m scripts.stale_decision_think --config CFG smoke --batch-size B --output DIR/smoke.json
    python -m scripts.stale_decision_think --config CFG freeze --output DIR/freeze.json
    python -m scripts.stale_decision_think --config CFG run --freeze DIR/freeze.json --batch-size B --output DIR/records.jsonl
    python -m scripts.stale_decision_think --config CFG analyze --records DIR/records.jsonl --output DIR/report.json

smoke: 16 items. It reports throughput, trace lengths, how many traces close, and how often batched and unbatched
greedy outputs agree, and projects the GPU-hours of the full item set. It writes no outcome statistics.
"""
import argparse
import json
import time
from pathlib import Path

from src.data.stale_decision_v2 import digest, write_new
from src.experiments import stale_decision_think as exp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('command', choices=['smoke', 'freeze', 'run', 'analyze'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--freeze')
    p.add_argument('--records')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    exp.check_config(config)
    if a.command == 'freeze':
        write_new(a.output, {'protocol': exp.PROTOCOL, 'code_hash': exp.code_hash(), 'config_hash': digest(config),
                             'status': 'frozen', 'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        return
    if a.command == 'analyze':
        records = [json.loads(l) for l in Path(a.records).read_text().splitlines()]
        out = exp.analyze(config, records)
        out.update(code_hash=exp.code_hash(), config_hash=digest(config), records_hash=digest(records))
        write_new(a.output, out)
        return
    model, tok = exp.load(config)
    if a.command == 'smoke':
        rows = exp.items(config)[:16]
        start = time.perf_counter()
        batched = exp.generate_batch(model, tok, config, rows)
        t_batch = time.perf_counter() - start
        single = [exp.generate_batch(model, tok, config, [r])[0] for r in rows[:4]]
        agree = sum(b['raw_text'] == s['raw_text'] for b, s in zip(batched[:4], single))
        n_all = len(exp.items(config))
        write_new(a.output, {'batch_size': len(rows), 'seconds': t_batch, 'seconds_per_item': t_batch / len(rows),
                             'projected_gpu_hours_all': t_batch / len(rows) * n_all / 3600, 'items_all': n_all,
                             'closed': sum(b['closed'] for b in batched), 'terminated': sum(b['terminated'] for b in batched),
                             'generated_tokens': [b['n_generated_tokens'] for b in batched],
                             'batched_vs_unbatched_identical': f'{agree}/4',
                             'samples': [{'final_text': b['final_text'], 'tail': b['raw_text'][-200:]} for b in batched[:4]],
                             'code_hash': exp.code_hash(), 'config_hash': digest(config)})
        print('smoke', round(t_batch / len(rows), 2), 's/item; projected', round(t_batch / len(rows) * n_all / 3600, 2),
              'GPU-h; closed', sum(b['closed'] for b in batched), '/16; identical', agree, '/4')
        return
    frozen = json.loads(Path(a.freeze).read_text())
    if frozen['code_hash'] != exp.code_hash() or frozen['config_hash'] != digest(config):
        raise ValueError('freeze differs from the active protocol')
    start = time.perf_counter()
    rows = exp.run(config, a.output, model, tok, a.batch_size)
    Path(a.output + '.timing.json').write_text(json.dumps({'items': len(rows), 'seconds': time.perf_counter() - start}))


if __name__ == '__main__':
    main()
