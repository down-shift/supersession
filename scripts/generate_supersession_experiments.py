#!/usr/bin/env python3
"""Generate explicit paired behavioral controls/status inputs or legacy chains."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from src.data.io import write_jsonl, read_jsonl, sha256_file
from src.data.supersession import make_version_chain, render_version_chain
from src.data.supersession_behavior import generate_behavior_pairs, generate_status_2x2_gate, SCHEMA
from src.utils import provenance, save_json


def _history_signature(row):
    return json.dumps({'values': row['matching_values'], 'entities': row.get('entities', row.get('variables')),
                       'attribute': row.get('attribute'), 'orientation': row.get('orientation')}, sort_keys=True)


def _recompute_gate_pass(gate):
    """Recompute the frozen gate from hash-bound raw scores; do not trust JSON pass/metrics."""
    from src.analysis.natural_competence import GATE, GATE_VERSION, evaluate_competence
    if gate.get('gate_version') != GATE_VERSION or gate.get('preregistered_gate') != GATE:
        return False
    if gate.get('competence_code_sha256') != sha256_file('src/analysis/natural_competence.py'):
        return False
    try:
        result = evaluate_competence(read_jsonl(gate['dataset_path']), read_jsonl(gate['behavior_path']))
    except (ValueError, KeyError, TypeError):
        return False
    return (result['pass'] and gate.get('dataset_seed') == result['dataset_seed'] and
            gate.get('expected_cell_count') == result['expected_cell_count'] and
            gate.get('by_template_condition_query_orientation_edit_status_pair_direction_slot') == result['summary'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('controls', 'status', 'controls_counterbalanced', 'status_2x2', 'status_2x2_gate', 'chains'), required=True)
    p.add_argument('--values', required=True, help='JSON array of candidate strings')
    p.add_argument('--token-ids', help='restrict controls/status proposals to this validated token map')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, help='number of histories (defaults: 96, or fixed 24 for status_2x2_gate)')
    p.add_argument('--seed', type=int)
    p.add_argument('--frozen-gate', help='passing frozen competence artifact required for new status_2x2 confirmatory data')
    p.add_argument('--depths', default='0,1,2,4,8')
    p.add_argument('--natural-language', action='store_true', help='render controlled entity–attribute prompts')
    p.add_argument('--templates', default='nora_v1', help='comma-separated natural template IDs; balance across histories')
    p.add_argument('--frozen-gate-required', help='confirmatory natural run requires a passing fresh competence gate JSON')
    p.add_argument('--prior-dataset', action='append', default=[], help='earlier development/gate dataset; confirmatory histories must be disjoint')
    p.add_argument('--config', help='scoring config, required to verify a frozen natural gate')
    p.add_argument('--stage', choices=('development', 'frozen_gate', 'confirmatory'), default='development')
    a = p.parse_args()
    if a.natural_language and a.kind != 'controls_counterbalanced':
        raise ValueError('--natural-language is supported only with --kind controls_counterbalanced')
    if a.natural_language and a.stage == 'confirmatory' and not a.frozen_gate_required:
        raise ValueError('natural confirmatory generation requires --frozen-gate-required')
    if a.natural_language and a.stage != 'confirmatory' and a.frozen_gate_required:
        raise ValueError('--frozen-gate-required is only valid for confirmatory natural generation')
    if a.kind == 'status_2x2':
        if not a.frozen_gate or not a.token_ids:
            raise ValueError('status_2x2 generation requires --frozen-gate and --token-ids; use generate_status_prompt_stage.py for redevelopment')
        from scripts.generate_status_prompt_stage import generate_stage
        generate_stage('confirmatory', a.values, a.token_ids, a.output, a.n, a.seed,
                       frozen_gate_path=a.frozen_gate)
        return
    gate = None
    if a.natural_language and a.frozen_gate_required:
        if not a.config or not a.token_ids:
            raise ValueError('natural confirmatory generation requires --config and --token-ids to verify the frozen gate')
        from src.data.supersession_behavior import NATURAL_TEMPLATES
        from src.utils import load_config
        gate_path = Path(a.frozen_gate_required)
        gate = json.loads(gate_path.read_text())
        claimed = gate.pop('artifact_sha256', None)
        actual = hashlib.sha256(json.dumps(gate, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if claimed != actual or gate.get('stage') != 'frozen_competence_gate' or gate.get('pass') is not True or not _recompute_gate_pass(gate):
            raise ValueError('frozen gate is failed or its integrity seal is invalid')
        gate['artifact_sha256'] = claimed
        load_config(a.config)
        renderer_hash = hashlib.sha256(Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()
        score_prov_path = Path(gate['behavior_path'] + '.provenance.json')
        if (gate['dataset_sha256'] != sha256_file(gate['dataset_path']) or gate['behavior_sha256'] != sha256_file(gate['behavior_path'])
                or gate['token_map_sha256'] != sha256_file(a.token_ids) or gate['config_sha256'] != sha256_file(a.config)
                or gate['renderer_sha256'] != renderer_hash or not score_prov_path.exists()
                or gate['scoring_provenance_sha256'] != sha256_file(score_prov_path)):
            raise ValueError('frozen gate provenance differs from dataset, scores, token map, config, or renderer')
        score_prov = json.loads(score_prov_path.read_text())
        for key in ('model_revision', 'tokenizer_revision'):
            if gate['model_tokenizer'].get(key) != score_prov.get(key):
                raise ValueError(f'frozen gate {key} does not match its scoring provenance')
        if a.templates not in gate['selected_templates'] or a.templates != gate['selected_template']:
            raise ValueError('confirmatory template differs from the frozen selected template')
        if not a.prior_dataset:
            raise ValueError('confirmatory generation requires --prior-dataset for development overlap auditing')
        prior_hashes = [sha256_file(path) for path in a.prior_dataset]
        if prior_hashes != gate.get('prior_dataset_sha256'):
            raise ValueError('development datasets differ from those used to qualify the frozen gate')
        seen = set()
        gate_signatures = set(gate['history_signatures'])
        for prior in a.prior_dataset:
            current = {_history_signature(row) for row in read_jsonl(prior)}
            if current & seen:
                raise ValueError('provided prior-stage datasets overlap each other')
            seen |= current
        if seen & gate_signatures:
            raise ValueError('frozen gate concrete histories overlap development histories')
    seed = a.seed if a.seed is not None else (20261001 if a.kind == 'status_2x2_gate' else 73021)
    n = a.n if a.n is not None else (24 if a.kind == 'status_2x2_gate' else 96)
    if a.kind == 'status_2x2_gate' and n != 24:
        raise ValueError('status_2x2_gate requires --n 24')
    out = Path(a.output)
    if out.exists() or Path(str(out) + '.provenance.json').exists():
        raise FileExistsError(f'{out} exists; choose a fresh output path')
    proposals = json.loads(Path(a.values).read_text())
    values = proposals
    if a.token_ids:
        if a.kind == 'chains':
            raise ValueError('--token-ids is supported only for controls/status generation')
        token_map = json.loads(Path(a.token_ids).read_text())['token_ids']
        values = [v for v in proposals if v in token_map]
        print(f'using {len(values)} validated proposal values; excluded {len(proposals) - len(values)}')
    if a.kind == 'status_2x2_gate':
        rows = generate_status_2x2_gate(n, values, seed)
    elif a.kind in ('controls', 'status', 'controls_counterbalanced', 'status_2x2'):
        rows = generate_behavior_pairs(a.kind, n, values, seed)
    else:
        rows = []
        for i in range(n):
            for depth in map(int, a.depths.split(',')):
                h = make_version_chain(f'chain{i:05d}:d{depth}', values, depth, seed+i*19+depth)
                h['rendered_prompt_x'] = render_version_chain(h, 'x')
                h['rendered_prompt_z'] = render_version_chain(h, 'z')
                rows.append(h)
    if a.natural_language:
        from src.data.supersession_behavior import NATURAL_TEMPLATES
        templates = [x.strip() for x in a.templates.split(',') if x.strip()]
        if not templates or set(templates) - NATURAL_TEMPLATES.keys():
            raise ValueError('unknown or empty natural template selection')
        names = ('Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah', 'Zoe', 'Theo', 'Maya', 'Leo')
        attrs = ('badge', 'color', 'code', 'label')
        expanded = []
        for row in rows:
            i = int(row['history_index'])
            for template in templates:
                clone = copy.deepcopy(row)
                old_hid = clone['history_id']
                clone['history_id'] = f'{old_hid}:{template}'
                clone['pair_id'] = clone['pair_id'].replace(old_hid, clone['history_id'], 1)
                clone['example_id'] = f'{clone["pair_id"]}:{clone["pair_direction"]}'
                clone.update(prompt_family='natural_entity_attribute_v1', prompt_variant=template,
                             entities=[names[(2*i) % len(names)], names[(2*i+1) % len(names)]],
                             attribute=attrs[i % len(attrs)])
                clone['variables'] = clone['entities'] if clone.get('orientation', 0) == 0 else clone['entities'][::-1]
                expanded.append(clone)
        rows = expanded
        if a.frozen_gate_required:
            prior_signatures = {_history_signature(row) for path in a.prior_dataset for row in read_jsonl(path)} | gate_signatures
            overlap = prior_signatures & {_history_signature(row) for row in rows}
            if overlap:
                raise ValueError('confirmatory histories overlap a development/frozen-gate dataset')
    write_jsonl(rows, out)
    save_json({**provenance({'seed': seed}, out), 'schema': SCHEMA if a.kind != 'chains' else 'legacy_chains',
               'kind': a.kind, 'n_matched_histories': n, 'n_records': len(rows),
               'values_sha256': sha256_file(a.values), 'token_map_sha256': sha256_file(a.token_ids) if a.token_ids else None,
               'selected_values': values, 'excluded_proposals': [v for v in proposals if v not in values],
               'prompt_family': 'natural_entity_attribute_v1' if a.natural_language else None,
               'stage': a.stage if a.natural_language else None,
               'templates': a.templates.split(',') if a.natural_language else None,
               'frozen_gate_sha256': sha256_file(a.frozen_gate_required) if a.natural_language and a.frozen_gate_required else None,
               'prior_dataset_sha256': [sha256_file(path) for path in a.prior_dataset] if a.natural_language else [],
               'config_sha256': sha256_file(a.config) if a.natural_language and a.config else None,
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/supersession_behavior.py').read_bytes()).hexdigest(),
               'update_block_order_balance': ({**__import__('src.data.supersession_behavior',fromlist=['audit_status_2x2_update_order']).audit_status_2x2_update_order(rows)} if a.kind in ('status_2x2','status_2x2_gate') else None)}, str(out) + '.provenance.json')
    print(f'wrote {len(rows)} records for {n} histories to {out}')


if __name__ == '__main__':
    main()
