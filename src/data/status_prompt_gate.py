"""Accuracy-only prompt development, independent order crossing, and frozen gates.

Legacy status datasets/renderers are deliberately preserved. New histories keep
literal x/z fixed and cross update order within each history, independently of
status, query, and wording. No edit effects enter prompt selection or gating.
"""
import copy
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.io import read_jsonl, sha256_file
from src.data.supersession_behavior import FIELDS, generate_behavior_pairs, edited_member

VARIANTS = ('bracketed', 'inline_status', 'action_labels')
STATUSES = ('YY', 'YN', 'NY', 'NN')
ORDERS = ('xz', 'zx')
STAGES = ('development', 'frozen_gate', 'confirmatory')
DEFAULT_SEEDS = {'development': 20261010, 'frozen_gate': 20261011, 'confirmatory': 20261012}
FORBIDDEN = {'pair_id', 'pair_direction', 'edited_field', 'source_value', 'replacement_value',
             'candidate_logits', 'candidate_probabilities', 'identity_transfer', 'matched_edit_effect',
             'source_logit_before', 'source_logit_after', 'replacement_logit_before', 'replacement_logit_after'}


def template_hash():
    # Bind approval to both the new templates and the shared answer/chat wrapper.
    return hashlib.sha256(Path(__file__).read_bytes() +
                          Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()


def prompt_text(row):
    values, variant = row['semantic_values'], row['prompt_variant']
    if variant not in VARIANTS or row['update_order'] not in ORDERS:
        raise ValueError('unsupported prompt variant/update order')
    lines = ['Initial state:', f"x = {values['initial_x']}", f"z = {values['initial_z']}", '', 'Updates:']
    for variable in row['update_order']:
        value = values[f'proposed_{variable}']
        accepted = row['update_accepted_by_variable'][variable]
        status = 'ACCEPTED' if accepted else 'REJECTED'
        if variant == 'bracketed':
            lines.append(f'{variable} -> {value} [{status}]')
        elif variant == 'inline_status':
            lines.append(f'Update {variable} = {value}; status: {status}.')
        else:
            lines.append(f'{"ACCEPT" if accepted else "REJECT"} update: {variable} = {value}')
    instruction = ('Apply only ACCEPTED updates. REJECTED updates do not change the initial state.'
                   if variant != 'action_labels' else
                   'Apply ACCEPT updates. Ignore REJECT updates; they do not change the initial state.')
    return '\n'.join(lines + ['', instruction, f'What is {row["query"]}?',
                              'Respond with only the value, with no explanation.'])


def history_signatures(rows):
    """Conservatively exclude both literal orientations of old concrete histories."""
    signatures = {tuple(r['matching_values'][f] for f in FIELDS) for r in rows}
    return signatures | {(s[1],s[0],s[3],s[2]) for s in signatures}


def generate_prompt_stage(stage, n, values, seed, variants=VARIANTS, excluded=(), frozen_gate_path=None):
    excluded = list(excluded)
    if stage == 'confirmatory':
        if not frozen_gate_path:
            raise ValueError('confirmatory generation requires an explicit passing frozen gate')
        gate = checked_artifact(frozen_gate_path, 'frozen_gate')
        gate_rows = read_jsonl(gate['dataset_path'])
        gate_generation = json.loads(Path(gate['dataset_path']+'.provenance.json').read_text())
        selection = checked_artifact(gate_generation['selection_path'], 'development')
        excluded += gate_rows + read_jsonl(selection['dataset_path'])
        if tuple(variants) != (gate['selected_variant'],) or list(values) != gate_rows[0]['candidate_values']:
            raise ValueError('final wording/candidate pool differs from frozen gate')
    if stage not in STAGES or (stage == 'frozen_gate' and n != 24) or (stage == 'confirmatory' and n != 96):
        raise ValueError('frozen_gate requires 24 histories; confirmatory requires 96')
    if n < 1 or len(set(values)) != len(values) or len(values) < 5:
        raise ValueError('positive n and at least five distinct values required')
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError('invalid/duplicate prompt variants')
    if stage != 'development' and len(variants) != 1:
        raise ValueError('gate/final must isolate exactly one selected prompt variant')
    if any(r.get('seed') == seed for r in excluded):
        raise ValueError('fresh stage must use a separate seed from excluded histories')
    seen = history_signatures(excluded)
    capacity = len(values)*(len(values)-1)*(len(values)-2)*(len(values)-3)
    available = capacity - sum(set(s) <= set(values) for s in seen)
    if n > available:
        raise ValueError('not enough fresh histories in candidate vocabulary')
    rng = random.Random(seed)
    rows = []
    for index in range(n):
        while True:
            chosen = tuple(rng.sample(list(values), 4))
            if chosen not in seen:
                seen.add(chosen)
                break
        # Reuse the explicit semantic status/answer derivation in the existing generator.
        # Its values are replaced below with this stage's fresh draw.
        contexts = generate_behavior_pairs('status_2x2', 2, values, seed + index)
        contexts = [r for r in contexts if r['history_index'] == 0 and r['pair_direction'] == 0]
        replacements = [v for v in values if v not in chosen]
        rng.shuffle(replacements)
        for variant, order in itertools.product(variants, ORDERS):
            for source in contexts:
                if stage != 'confirmatory' and source['edited_field'] != 'initial_x':
                    continue
                row = copy.deepcopy(source)
                hid = f'status2x2_{stage}_s{seed}_{index:06d}'
                original = dict(zip(FIELDS, chosen))
                row.update(history_id=hid, history_index=index, seed=seed, variables=['x','z'], orientation=0,
                           matching_values=original, semantic_values=copy.deepcopy(original),
                           prompt_variant=variant, update_order=order, prompt_stage=stage,
                           record_type='status_2x2_prompt_'+stage,
                           replacement_values={f: replacements[0 if f.endswith('x') else 1 % len(replacements)] for f in FIELDS})
                row['current_x'] = original[row['current_fields']['x']]
                row['current_z'] = original[row['current_fields']['z']]
                row['answer'] = row[f'current_{row["query"]}']
                row['roles'] = {'target':row['answer']}
                obsolete = row['obsolete_fields'].get(row['query'])
                row['stale_value'] = original[obsolete] if obsolete else None
                suffix = f'{variant}:{row["condition"]}:{row["query"]}:order_{order}'
                if stage == 'confirmatory':
                    field = row['edited_field']
                    row.update(pair_id=f'{hid}:{suffix}:{field}', pair_direction=0,
                               source_value=original[field], replacement_value=row['replacement_values'][field])
                    row['example_id'] = row['pair_id'] + ':0'
                    rows.extend((row, edited_member(row)))
                else:
                    for key in ('pair_id','pair_direction','edited_field','edited_variable','edit_status',
                                'source_value','replacement_value','replacement_values','unassigned_slot_order'):
                        row.pop(key, None)
                    row['example_id'] = f'{hid}:{suffix}'
                    rows.append(row)
    if stage == 'confirmatory':
        audit_final_dataset(rows)
    else:
        audit_prompt_dataset(rows)
    return rows


def audit_prompt_dataset(rows):
    if not rows:
        raise ValueError('empty prompt-development/gate dataset')
    stages = {r.get('prompt_stage') for r in rows}
    if len(stages) != 1 or next(iter(stages)) not in ('development', 'frozen_gate'):
        raise ValueError('mixed/unsupported prompt stages')
    stage = next(iter(stages))
    variants = {r.get('prompt_variant') for r in rows}
    if not variants <= set(VARIANTS) or (stage == 'frozen_gate' and len(variants) != 1):
        raise ValueError('gate requires one isolated prompt variant')
    histories, ids = defaultdict(dict), set()
    for row in rows:
        if FORBIDDEN & row.keys():
            raise ValueError('prompt selection/gate prohibits causal/edit fields')
        if row.get('record_type') != 'status_2x2_prompt_'+stage or row.get('experiment_kind') != 'status_2x2':
            raise ValueError('mixed prompt dataset record types')
        if row.get('condition') not in STATUSES or row.get('query') not in ('x','z') or row.get('update_order') not in ORDERS:
            raise ValueError('invalid status/query/update-order cell')
        if row.get('variables') != ['x','z'] or row.get('orientation') != 0:
            raise ValueError('new prompt roles must equal literal x/z independently of order')
        accepted = {v:row['condition'][i] == 'Y' for i,v in enumerate('xz')}
        current = {v:f'{"proposed" if accepted[v] else "initial"}_{v}' for v in 'xz'}
        semantic_status = {f:(('accepted_current' if f.startswith('proposed') else 'superseded_initial')
                               if accepted[f[-1]] else ('rejected_update' if f.startswith('proposed') else 'accepted_current'))
                           for f in FIELDS}
        obsolete = {v:f'initial_{v}' for v in 'xz' if accepted[v]}
        if (row.get('update_accepted_by_variable') != accepted or row.get('current_fields') != current
                or row.get('semantic_status') != semantic_status or row.get('obsolete_fields') != obsolete
                or row.get('status') != row['condition']
                or row.get('status_by_variable') != {v:('accepted' if accepted[v] else 'rejected') for v in 'xz'}):
            raise ValueError('semantic status/current binding mismatch')
        original = row['matching_values']
        if set(original) != set(FIELDS) or len(set(original.values())) != 4 or row['semantic_values'] != original:
            raise ValueError('gate must contain four distinct unedited semantic values')
        if not set(original.values()) <= set(row['candidate_values']):
            raise ValueError('missing candidate value')
        answer = original[current[row['query']]]
        if row['answer'] != answer or row['roles'] != {'target':answer}:
            raise ValueError('incorrect task answer metadata')
        for v in 'xz':
            if row[f'current_{v}'] != original[current[v]]:
                raise ValueError('incorrect current value metadata')
        if row['stale_value'] != (original[obsolete[row['query']]] if row['query'] in obsolete else None):
            raise ValueError('incorrect obsolete metadata')
        key = (row['prompt_variant'], row['condition'], row['query'], row['update_order'])
        if row['example_id'] in ids or key in histories[row['history_id']]:
            raise ValueError('duplicate prompt example/cell')
        ids.add(row['example_id']); histories[row['history_id']][key] = row
    required = set(itertools.product(variants, STATUSES, 'xz', ORDERS))
    if stage == 'frozen_gate' and len(histories) != 24:
        raise ValueError('frozen gate requires exactly 24 fresh histories')
    signatures = set()
    for cells in histories.values():
        if set(cells) != required:
            raise ValueError('missing required balanced variant/status/query/order cells')
        ref = next(iter(cells.values()))
        for row in cells.values():
            for field in ('matching_values','candidate_values','seed','history_index','variables','orientation'):
                if row[field] != ref[field]:
                    raise ValueError('prompt variants/status/query/order must share identical history context')
        signature = tuple(ref['matching_values'][f] for f in FIELDS)
        if signature in signatures:
            raise ValueError('duplicate concrete prompt histories')
        signatures.add(signature)
    if len({r['seed'] for r in rows}) != 1:
        raise ValueError('mixed stage seeds')
    return stage


def audit_final_dataset(rows):
    """Audit each independently crossed order through the unchanged pair audit."""
    from src.data.supersession_behavior import audit_behavior_dataset
    if not rows or {r.get('prompt_stage') for r in rows} != {'confirmatory'}:
        raise ValueError('invalid final dataset stage')
    variants = {r.get('prompt_variant') for r in rows}
    if len(variants) != 1 or not variants <= set(VARIANTS):
        raise ValueError('final dataset must isolate selected wording')
    if len({r['history_id'] for r in rows}) != 96:
        raise ValueError('final dataset requires 96 fresh histories')
    if {r.get('update_order') for r in rows} != set(ORDERS):
        raise ValueError('final dataset requires both update orders')
    for row in rows:
        if row.get('record_type') != 'status_2x2_prompt_confirmatory' or row['condition'] not in STATUSES:
            raise ValueError('mixed/invalid final prompt records')
        accepted = {v:row['condition'][i]=='Y' for i,v in enumerate('xz')}
        if row.get('update_accepted_by_variable') != accepted or row.get('variables') != ['x','z']:
            raise ValueError('final update semantics/literal variables mismatch')
        pid = (f'{row["history_id"]}:{row["prompt_variant"]}:{row["condition"]}:{row["query"]}'
               f':order_{row["update_order"]}:{row["edited_field"]}')
        if row['pair_id'] != pid or row['example_id'] != f'{pid}:{row["pair_direction"]}':
            raise ValueError('final pair IDs disagree with explicit order/variant metadata')
    for order in ORDERS:
        normalized = []
        for row in rows:
            if row['update_order'] != order:
                continue
            r = copy.deepcopy(row)
            for key in ('prompt_variant','prompt_stage','record_type','update_order'):
                r.pop(key)
            pid = f'{r["history_id"]}:{r["condition"]}:{r["edited_field"]}:current_{r["query"]}'
            r['pair_id'] = pid; r['example_id'] = f'{pid}:{r["pair_direction"]}'
            normalized.append(r)
        audit_behavior_dataset(normalized, 'status_2x2')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['history_id'],row['condition'],row['edited_field'],row['query'],row['pair_direction'])].append(row)
    for members in grouped.values():
        if len(members) != 2:
            raise ValueError('final cells missing independently crossed order')
        a,b = members
        if {k:v for k,v in a.items() if k not in ('update_order','pair_id','example_id')} != {k:v for k,v in b.items() if k not in ('update_order','pair_id','example_id')}:
            raise ValueError('update order changes semantic content')
    return 'status_2x2'


def competence_summary(dataset, scores):
    stage = audit_prompt_dataset(dataset)
    expected = {r['example_id']:r for r in dataset}
    if len(scores) != len(expected) or {r.get('example_id') for r in scores} != set(expected):
        raise ValueError('scores must exactly cover all balanced task cells')
    groups = defaultdict(list)
    for score in scores:
        if FORBIDDEN & score.keys():
            raise ValueError('causal/effect data prohibited in prompt selection')
        if any(score.get(k) != v for k,v in expected[score['example_id']].items()):
            raise ValueError('scored semantic metadata differs from prompt dataset')
        accuracy, rank = score.get('full_vocab_next_token_accuracy'), score.get('target_rank')
        if accuracy not in (0,1) or type(rank) is not int or rank < 1 or not isinstance(score.get('prompt'), str):
            raise ValueError('missing/invalid accuracy, rank, or rendered prompt')
        if accuracy == 1 and rank != 1:
            raise ValueError('correct answer must have target rank 1')
        groups[(score['prompt_variant'],score['condition'],score['query'],score['update_order'])].append(score)
    cells = []
    for (variant,status,query,order), members in sorted(groups.items()):
        cells.append({'prompt_variant':variant,'status':status,'query':query,'update_order':order,
                      'n_histories':len(members), 'full_vocab_accuracy':sum(r['full_vocab_next_token_accuracy'] for r in members)/len(members),
                      'target_rank_mean':sum(r['target_rank'] for r in members)/len(members),
                      'target_rank_max':max(r['target_rank'] for r in members)})
    passed = all(c['full_vocab_accuracy'] == 1 and c['target_rank_max'] == 1 for c in cells)
    return {'stage':stage,'cells':cells,'gate_pass':passed,'threshold':1.0,
            'failure_rule':'any error in any status/query/update_order cell fails the frozen gate',
            'causal_R_inspected':False,'template_sha256':template_hash()}


def checked_artifact(path, required_stage):
    """Recompute competence from immutable inputs; never trust a gate boolean alone."""
    artifact = json.loads(Path(path).read_text())
    if artifact.get('stage') != required_stage or artifact.get('template_sha256') != template_hash():
        raise ValueError('wrong stage or changed prompt template in gate/selection artifact')
    for key in ('dataset','scores'):
        if sha256_file(artifact[key+'_path']) != artifact[key+'_sha256']:
            raise ValueError('gate/selection input hash changed')
    dataset = read_jsonl(artifact['dataset_path']); scores = read_jsonl(artifact['scores_path'])
    recomputed = competence_summary(dataset, scores)
    if recomputed['cells'] != artifact.get('cells') or recomputed['gate_pass'] != artifact.get('gate_pass'):
        raise ValueError('gate artifact does not match recomputed competence')
    variant = artifact.get('selected_variant')
    if variant not in {r['prompt_variant'] for r in dataset}:
        raise ValueError('selected variant missing from competence dataset')
    generation_path = artifact['dataset_path']+'.provenance.json'
    if sha256_file(generation_path) != artifact.get('generation_provenance_sha256'):
        raise ValueError('generation provenance changed')
    generation = json.loads(Path(generation_path).read_text())
    if generation.get('stage') != required_stage or generation.get('template_sha256') != template_hash():
        raise ValueError('missing/mismatched stage generation provenance')
    score_prov_path = artifact['scores_path'] + '.provenance.json'
    if sha256_file(score_prov_path) != artifact.get('scoring_provenance_sha256'):
        raise ValueError('scoring provenance changed')
    scoring = json.loads(Path(score_prov_path).read_text())
    if (scoring.get('dataset_sha256') != artifact['dataset_sha256'] or not scoring.get('token_map_sha256')
            or not scoring.get('config_sha256') or not scoring.get('model_id')
            or scoring.get('causal_effects_computed') is not False):
        raise ValueError('missing/mismatched competence-scoring provenance')
    if generation.get('dataset_sha256') != artifact['dataset_sha256'] or generation.get('token_map_sha256') != scoring['token_map_sha256']:
        raise ValueError('generation dataset/token map differs from scoring')
    import re
    for key in ('model_revision','tokenizer_revision'):
        if not re.fullmatch('[0-9a-f]{40}',str(scoring.get(key))):
            raise ValueError('frozen scoring must identify exact model/tokenizer revisions')
    if required_stage == 'frozen_gate':
        if not recomputed['gate_pass'] or artifact.get('threshold') != 1.0:
            raise ValueError('frozen competence gate failed; confirmatory generation forbidden')
        selection = checked_artifact(generation['selection_path'], 'development')
        if sha256_file(generation['selection_path']) != generation['selection_sha256']:
            raise ValueError('selection artifact changed after frozen gate')
        if selection['selected_variant'] != variant:
            raise ValueError('gate wording differs from development selection')
        development = read_jsonl(selection['dataset_path'])
        if history_signatures(dataset) & history_signatures(development) or dataset[0]['seed'] == development[0]['seed']:
            raise ValueError('frozen gate not fresh/disjoint from prompt development')
        dev_scoring = json.loads(Path(selection['scores_path']+'.provenance.json').read_text())
        for key in ('model_id','model_revision','tokenizer_revision','config_sha256','token_map_sha256'):
            if scoring[key] != dev_scoring[key]:
                raise ValueError('gate model/config/token-map differs from prompt development')
    return artifact
