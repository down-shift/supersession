"""Separate frozen contract, competence decision and provenance for relational v2."""
import copy
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

from src.cross_model import protocol as v1
from src.cross_model.robustness_v2 import (
ALLOCATION_TABLES, ALT_RELATIONS, ATTRIBUTES, CONDITIONS, COUNTS, DESIGN_REVISION, ENTITY_PAIRS, FRACTIONAL_RELATION_ONE, RENDERING, ROWS_PER_HISTORY,
    SEEDS, VERSION, VALUES, EXCLUSION_LEDGER, concrete_signature, disjoint, generate, validate_rows)
from src.cross_model.score_checks import checked_scores
from src.cross_model.workflow import local_artifact_path
from src.data.io import read_jsonl, sha256_file
from src.utils import provenance

CAUSAL_VERSION = v1.CAUSAL_VERSION
GATE = {'scope': 'unique prompts within each of six focal conditions; full factorial cells diagnostic',
        'semantic_accuracy_min': .99, 'candidate_rank_policy': 'strict rank one; ties incorrect; mean rank diagnostic only',
        'mean_current_minus_stale_min': 0.0, 'stale_margin_conditions': ['superseded']}
CONTRACT = {'protocol': VERSION, 'design_revision': DESIGN_REVISION, 'causal_protocol': CAUSAL_VERSION,
            'token_geometry_schema': 2,
            'values': VALUES, 'seeds': SEEDS, 'counts': COUNTS, 'validation_histories': 4,
            'conditions': list(CONDITIONS), 'gate': GATE, 'template': 'nora_relational_v2',
            'rendering': RENDERING, 'attributes': list(ATTRIBUTES),
            'entity_pairs': [list(pair) for pair in ENTITY_PAIRS],
            'alternate_relations': list(ALT_RELATIONS),
            'fractional_relation_one_cells': [list(cell) for cell in FRACTIONAL_RELATION_ONE],
            'allocation_tables': {stage: [list(cell) for cell in cells] for stage, cells in ALLOCATION_TABLES.items()},
            'value_allocation': 'four distinct historical/current assignment values plus two distinct replacement identities outside all four assignments',
            'freshness_policy': 'fixed prior-history exclusion ledger; structural rejection within fixed seeded streams in validation/development/gate/confirmation priority; no seed retries or outcome-based exclusions',
            'order_factors': ['historical_entity_or_unassigned_mention_order', 'current_entity_order'],
            'live_order_policy': 'one historical/live assignment block; current-order labels are duplicate nuisance variants, not a physical second block',
            'pair_direction': '0 baseline; 1 edited; independent x and z interventions; no activation transfers or reciprocal swaps',
            'rows_per_history': ROWS_PER_HISTORY, 'surfaces': v1.CONTRACT['surfaces'],
            'candidate_score': v1.CONTRACT['candidate_score'],
            'development_policy': 'competence/implementation only; no causal report or prompt/vocabulary redevelopment',
            'span_policy': v1.CONTRACT['span_policy'],
            'surface_geometry_audit': v1.CONTRACT['surface_geometry_audit'],
            'primary_contrasts': ['R_superseded - R_early_unassigned', 'R_superseded - R_entity_mention', 'R_superseded - R_other_attribute'],
            'other_attribute_label': 'different_relation_control; team/project relations are counterbalanced, not claimed unrelated',
            'secondary': ['R_superseded - R_late_unassigned', 'R_live - R_superseded',
                          'live-minus-each-control', 'aligned-versus-reversed historical/current order'],
            'aggregation': 'mean E over four order cells within each history/condition/edit/query before symmetric relevance; aligned/reversed subsets each average their two cells; contrasts and bootstrap on histories',
            'confirmation_policy': 'all trials; primary estimates never conditioned on correctness',
            'bootstrap': v1.CONTRACT['bootstrap']}
MODEL_SETTINGS = {
    'qwen3_8b': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218', 'float16'),
    'gemma3_4b': ('google/gemma-3-4b-it', '093f9f388b31de276ce2de164bdc2081324b9767', 'bfloat16'),
    'phi4_mini': ('microsoft/Phi-4-mini-instruct', 'cfbefacb99257ffa30c83adab238a50856ac3083', 'float16')}


def model_slug(config):
    return next((slug for slug, (model_id, _, _) in MODEL_SETTINGS.items()
                 if model_id == config['model']['id']), None)


def validate_config(config):
    slug = model_slug(config)
    if slug is None or config.get('cross_model') != CONTRACT:
        raise ValueError('exact relational v2 model panel and frozen contract required')
    m = config['model']
    model_id, revision, dtype = MODEL_SETTINGS[slug]
    if (m.get('revision') != revision or m.get('tokenizer_revision') != revision
            or (m.get('tokenizer_id') or m['id']) != model_id or m.get('dtype') != dtype):
        raise ValueError('model/tokenizer revisions or dtype differ from the frozen original v1 run')
    if (not m.get('chat_template') or m.get('attn_implementation') != 'eager' or
            m.get('quantization') != 'int8' or not m.get('require_full_gpu') or m.get('device_map') != 0):
        raise ValueError('v2 requires existing eager/chat/int8/full-GPU settings')
    if slug == 'phi4_mini' and not m.get('trust_remote_code'):
        raise ValueError('pinned Phi implementation requires the existing remote-code setting')


def verify_v1_lineage(config, path=None):
    """Compare with the actual saved score provenance, including Gemma's actual pin."""
    validate_config(config)
    path = Path(path or f'outputs/{model_slug(config)}_review2/confirmatory_scores.jsonl.provenance.json')
    saved = v1.read_sealed(path)
    original = saved['provenance']
    if saved.get('stage') != 'confirmatory' or original.get('protocol') != 'cross_model_v1':
        raise ValueError('continuity requires original cross_model_v1 confirmation provenance')
    expected = {'model_id': config['model']['id'], 'model_revision': config['model']['revision'],
                'tokenizer_id': config['model'].get('tokenizer_id') or config['model']['id'],
                'tokenizer_revision': config['model']['tokenizer_revision'],
                'dtype': config['model']['dtype'], 'quantization': config['model']['quantization']}
    for key, value in expected.items():
        if original.get(key) != value:
            raise ValueError(f'actual original v1 provenance mismatch: {key}')
    for key in ('attn_implementation', 'chat_template', 'trust_remote_code'):
        default = False if key == 'trust_remote_code' else None
        if original['config']['model'].get(key, default) != config['model'].get(key, default):
            raise ValueError(f'actual original v1 setting mismatch: {key}')
    score_path = Path(str(path).removesuffix('.provenance.json'))
    if sha256_file(score_path) != saved['scores_sha256']:
        raise ValueError('original v1 score hash differs from provenance')
    candidate_path = path.parent/'candidates.json'
    if not candidate_path.exists() or sha256_file(candidate_path) != original['candidate_map_sha256']:
        raise ValueError('original v1 candidate map missing or hash differs from actual run provenance')
    candidate = v1.read_sealed(candidate_path)
    return {'path': str(path.resolve()), 'sha256': sha256_file(path), 'scientific_settings': expected,
            'original_scores_sha256': saved['scores_sha256'],
            'candidate_map_path': str(candidate_path.resolve()), 'candidate_map_sha256': sha256_file(candidate_path),
            'events_sha256': v1.digest(candidate['events']),
            'chat_template_sha256': original['chat_template_sha256'],
            'tokenizer_sha256': original['tokenizer_sha256']}


def code_hash():
    paths = sorted(Path('src/cross_model').glob('*.py')) + [
        Path('src/data/supersession_behavior.py'), Path('src/analysis/supersession_behavior.py'),
        Path('src/analysis/metrics.py'), Path('src/models/loader.py'), Path('src/utils.py'),
        Path('src/data/progress.py'), Path('scripts/robustness_v2.py')]
    paths.append(Path('scripts/run_relational_robustness.sh'))
    paths.append(Path('configs/cross_model_relational_v2_geometryfix_exclusions/prior_history_exclusions.json'))
    return v1.digest({str(p): sha256_file(p) for p in paths})


def manifest(config, config_path, candidate_path=None, dataset_path=None, **_):
    validate_config(config)
    result = provenance(config, dataset_path)
    result['config'] = copy.deepcopy(config)
    result['config'].pop('transformers_compatibility_shims', None)
    for key in list(result['config']):
        if key.startswith('resolved_'):
            result['config'].pop(key)
    result.update(protocol=VERSION, causal_protocol=CAUSAL_VERSION, contract_sha256=v1.digest(CONTRACT),
                  config_sha256=sha256_file(config_path), code_sha256=code_hash(),
                  renderer_sha256=sha256_file('src/cross_model/robustness_v2.py'),
                  candidate_map_sha256=sha256_file(candidate_path) if candidate_path else None,
                  prompt_template=RENDERING, scoring_config_seed=config.get('seed'),
                  dataset_seed=sorted({r['seed'] for r in read_jsonl(dataset_path)}) if dataset_path else None)
    return result


def check_manifest(saved, config, config_path, candidate_path=None, dataset_path=None, **_):
    current = manifest(config, config_path, candidate_path, dataset_path)
    for key in ('protocol', 'causal_protocol', 'contract_sha256', 'renderer_sha256', 'candidate_map_sha256',
                'dataset_sha256', 'model_id', 'model_revision', 'tokenizer_id', 'tokenizer_revision', 'dtype', 'quantization'):
        if saved.get(key) != current.get(key):
            raise ValueError(f'relational provenance mismatch: {key}')
    saved_config = copy.deepcopy(saved['config'])
    saved_config.pop('transformers_compatibility_shims', None)
    for key in list(saved_config):
        if key.startswith('resolved_'):
            saved_config.pop(key)
    if saved_config != current['config']:
        raise ValueError('relational scientific config mismatch')


def validate_dataset(rows, stage):
    return validate_rows(rows, stage)


def _metrics(entries):
    count = len(entries)
    margins = [m for _, _, m in entries if m is not None]
    return {'unique_prompts': count, 'semantic_accuracy': sum(correct for correct, _, _ in entries)/count,
            'mean_semantic_rank': sum(rank for _, rank, _ in entries)/count,
            'mean_current_minus_stale': sum(margins)/len(margins) if margins else None}


def evaluate(rows, scores, *, gate=True):
    if not rows or rows[0]['cross_model_stage'] not in ('development', 'frozen_gate'):
        raise ValueError('competence evaluation requires development or gate rows')
    validate_dataset(rows, rows[0]['cross_model_stage'])
    actual, _, ranks = checked_scores(rows, scores)
    cells, conditions, prompts = defaultdict(dict), defaultdict(dict), {}
    for row in rows:
        score = actual[row['example_id']]
        if score.get('score_kind') != 'competence_only' or 'matched_edit_effect' in score:
            raise ValueError('development/gate must use competence-only saved scores')
        masses, answer = score['semantic_log_mass'], row['answer']
        margin = masses[answer]-masses[row['stale_value']] if row['stale_value'] else None
        diagnostic = (int(ranks[row['example_id']] == 1), ranks[row['example_id']], margin)
        signature = score['prompt']
        if signature in prompts and prompts[signature] != (answer, masses):
            raise ValueError('inconsistent duplicate prompt scoring')
        prompts[signature] = (answer, masses)
        conditions[row['condition']][signature] = diagnostic
        cell = tuple(row[k] for k in ('condition', 'historical_entity_order', 'current_entity_order',
                                      'orientation', 'edited_variable', 'query', 'pair_direction'))
        cells[cell][signature] = diagnostic
    expected = set(product(CONDITIONS, (0, 1), (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)))
    if set(cells) != expected:
        raise ValueError('incomplete expected competence diagnostic cell product')
    by_condition = {c: _metrics(list(conditions[c].values())) for c in CONDITIONS}
    failures = []
    for condition, metrics in by_condition.items():
        reasons = []
        if metrics['semantic_accuracy'] < .99:
            reasons.append('semantic_accuracy')
        if condition == 'superseded' and metrics['mean_current_minus_stale'] <= 0:
            reasons.append('current_minus_stale')
        if reasons:
            failures.append({'condition': condition, 'reasons': reasons})
    return {'pass': not failures if gate else None,
            'eligibility_decision': 'frozen_gate' if gate else 'descriptive_only',
            'condition_summary': by_condition, 'failed_conditions': failures if gate else [],
            'cells': {'|'.join(map(str, k)): _metrics(list(v.values())) for k, v in sorted(cells.items())},
            'expected_cell_count': len(expected), 'unique_prompts': len(prompts),
            'rank_policy': 'recomputed strict rank; stored diagnostics checked; ties incorrect'}


def discover_prior_datasets(stage, roots=('outputs',)):
    """Collect available entity/attribute history files, including rejected previews."""
    result = []
    for root in map(Path, roots):
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.suffix not in ('.jsonl', '.json') or not path.is_file():
                continue
            name = path.name
            if any(word in name for word in ('scores', 'records', 'mechanism', 'analysis', 'report', 'audit', 'provenance', 'run.json')):
                continue
            if not any(word in name for word in ('development', 'confirmatory', 'gate', '_dev', 'preview')):
                continue
            if path.suffix == '.jsonl':
                with path.open() as handle:
                    line = next((line for line in handle if line.strip()), '')
                first = json.loads(line) if line else {}
            else:
                data = json.loads(path.read_text())
                first = next(iter(data.get('rows', [])), {}) if isinstance(data, dict) else {}
            if not first.get('matching_values') or not first.get('attribute') or not first.get('entities'):
                continue
            # Same corrected stage across models is intentionally shared. One-run
            # per-model claims separately prevent same-model retries/new paths.
            if first.get('design_revision') == DESIGN_REVISION and first.get('cross_model_stage') == stage:
                continue
            concrete_signature(first)  # Fail clearly on an unreconstructable eligible dataset.
            result.append(str(path))
    return result


def verify_freeze(path):
    frozen = v1.read_sealed(local_artifact_path(path))
    if frozen.get('contract') != CONTRACT or frozen.get('code_sha256') != code_hash():
        raise ValueError('frozen design or implementation changed; preserve the branch and freeze a new version')
    for entry in frozen['configs'].values():
        if sha256_file(entry['path']) != entry['sha256']:
            raise ValueError('frozen model config changed')
    audit_path = local_artifact_path(frozen['prompt_audit_path'])
    if sha256_file(audit_path) != frozen['prompt_audit_sha256']:
        raise ValueError('reviewed prompt audit changed after design freeze')
    return frozen


def verify_dataset_info(rows, info):
    freeze_path = local_artifact_path(info['freeze_path'])
    verify_freeze(freeze_path)
    if sha256_file(freeze_path) != info['freeze_sha256']:
        raise ValueError('dataset freeze binding changed')
    geometry_path = local_artifact_path(info['geometry_path'])
    if sha256_file(geometry_path) != info['geometry_sha256']:
        raise ValueError('dataset value/entity token geometry changed')
    geometry = v1.read_sealed(geometry_path)
    verify_geometry_rows(rows, geometry)
    registry = info['shared_history_registry']
    registry_path = local_artifact_path(registry['path'])
    if sha256_file(registry_path) != registry['sha256']:
        raise ValueError('shared-history registry hash changed')
    saved = v1.read_sealed(registry_path)
    if (saved['stage'] != info['stage'] or saved['contract_sha256'] != v1.digest(CONTRACT)
            or {h['concrete_signature'] for h in saved['histories']} != {concrete_signature(r) for r in rows}):
        raise ValueError('dataset differs from shared model-independent histories')


def verify_geometry_rows(rows, geometry):
    """Validate deduplicated physical spans against each row's semantic labels."""
    if set(geometry['example_to_prompt_sha256']) != {r['example_id'] for r in rows}:
        raise ValueError('token geometry does not cover every dataset member')
    if set(geometry.get('example_to_span_key', {})) != {r['example_id'] for r in rows}:
        raise ValueError('token geometry does not map semantic spans for every dataset member')
    for row in rows:
        prompt_hash = geometry['example_to_prompt_sha256'][row['example_id']]
        entry = geometry['by_prompt_sha256'][prompt_hash]
        spans = entry['spans']
        span_map = geometry['example_to_span_key'][row['example_id']]
        if entry['prompt_sha256'] != prompt_hash:
            raise ValueError('geometry prompt hash does not match its index')
        if (set(span_map) != set(spans)
                or not set(row['semantic_values']) <= set(span_map)
                or 'queried_entity' not in span_map):
            raise ValueError('token geometry semantic span map has missing or extraneous fields')
        if any(key not in spans for key in span_map.values()):
            raise ValueError('token geometry omits a relevant value/entity span')
        if any(s['token_length'] <= 0 or s['token_end_exclusive']-s['token_start'] != s['token_length']
               or len(s['token_ids']) != s['token_length'] or s['token_start'] < 0
               or s['token_end_exclusive'] > entry['prompt_token_length']
               for s in spans.values()):
            raise ValueError('invalid token geometry offsets/lengths')
        if (any(spans[span_map[f]]['text'] != value for f, value in row['semantic_values'].items())
                or spans[span_map['queried_entity']]['text'] != row['query_entity']):
            raise ValueError('geometry span texts differ from semantic values/entities')
    if set(geometry['by_prompt_sha256']) != set(geometry['example_to_prompt_sha256'].values()):
        raise ValueError('token geometry contains missing or extraneous prompts')
