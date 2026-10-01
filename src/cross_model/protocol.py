"""Immutable design constants and fail-closed artifact contracts."""
import hashlib
import json
import math
import re
from pathlib import Path

from src.analysis.natural_competence import concrete_history_signature, expected_cells
from src.data.io import read_jsonl, sha256_file
from src.data.supersession_behavior import audit_behavior_dataset, render_behavior_example
from src.utils import provenance

VERSION = 'cross_model_v1'
CAUSAL_VERSION = 'cross_model_sequence_mass_v1'
VALUES = ['amber', 'coral', 'jade', 'pearl', 'slate', 'teal', 'violet', 'ivory']
GATE = {'semantic_accuracy_min': .99, 'mean_semantic_rank_max': 1.01,
        'mean_current_minus_stale_min': 0.0}
SEEDS = {'validation': 20261201, 'development': 20261202,
         'frozen_gate': 20261203, 'confirmatory': 20261204}
COUNTS = {'development': 24, 'frozen_gate': 24, 'confirmatory': 96}
CONTRACT = {'protocol': VERSION, 'causal_protocol': CAUSAL_VERSION, 'values': VALUES,
            'gate': GATE, 'seeds': SEEDS, 'counts': COUNTS, 'template': 'nora_v1',
            'surfaces': 'lower/title x zero/one leading ASCII space; exclude unspaced prefix-changing forms; deduplicate token events',
            'candidate_score': 'logsumexp of complete unnormalized sequence log probabilities',
            'span_policy': 'patch all tokens; require exact pair alignment outside edited span',
            'bootstrap': {'unit': 'history', 'draws': 2000, 'seed': 73021},
            'mechanism_histories': 12, 'head_discovery_histories': 6,
            'depth_shift': {'early_max': .25, 'late_min': .75,
                            'contrasts': ['early_historical_minus_readout', 'late_readout_minus_historical']},
            'normalized_secondary': 'primary / live mean; available iff live bootstrap lower CI > 1 nat',
            'quantization_sensitivity': {'models': ['Qwen/Qwen3-8B', 'microsoft/Phi-4-mini-instruct'],
                                        'histories': 6, 'comparison': 'int8 versus unquantized float16',
                                        'patch_depths': [.125, .875], 'sites': ['edited_value_span','final_preanswer']}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def code_hash():
    paths = sorted(Path('src/cross_model').glob('*.py')) + [
        Path('scripts/cross_model.py'), Path('src/models/loader.py'), Path('src/data/supersession_behavior.py'),
        Path('src/analysis/natural_competence.py'), Path('src/analysis/supersession_behavior.py'),
        Path('src/analysis/metrics.py'), Path('src/utils.py'), Path('src/data/progress.py')]
    return digest({str(p): sha256_file(p) for p in paths})


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sealed(value):
    return {**value, 'seal': digest(value)}


def read_sealed(path):
    value = json.loads(Path(path).read_text())
    seal = value.pop('seal', None)
    if seal != digest(value):
        raise ValueError('artifact seal mismatch')
    return value


def validate_config(config):
    m = config['model']
    for key in ('revision', 'tokenizer_revision'):
        if not re.fullmatch('[0-9a-f]{40}', str(m.get(key))):
            raise ValueError(f'immutable {key} required')
    if not m.get('chat_template') or m.get('attn_implementation') != 'eager':
        raise ValueError('cross_model_v1 requires chat template and eager attention')
    if config.get('cross_model') != CONTRACT:
        raise ValueError('config must contain the exact cross_model_v1 contract')


def manifest(config, config_path, candidate_path=None, dataset_path=None):
    validate_config(config)
    result = provenance(config, dataset_path)
    from src.data.supersession_behavior import NATURAL_TEMPLATES
    result.update(prompt_template=list(NATURAL_TEMPLATES['nora_v1']),
                  scoring_config_seed=config.get('seed'),
                  dataset_seed=sorted({r['seed'] for r in read_jsonl(dataset_path)}) if dataset_path else None)
    result.update(protocol=VERSION, causal_protocol=CAUSAL_VERSION, contract_sha256=digest(CONTRACT),
                  config_sha256=sha256_file(config_path), code_sha256=code_hash(),
                  renderer_sha256=sha256_file('src/data/supersession_behavior.py'),
                  candidate_map_sha256=sha256_file(candidate_path) if candidate_path else None)
    return result


def check_manifest(saved, config, config_path, candidate_path=None, dataset_path=None):
    current = manifest(config, config_path, candidate_path, dataset_path)
    for k in ('protocol', 'causal_protocol', 'contract_sha256', 'config_sha256', 'code_sha256',
              'renderer_sha256', 'candidate_map_sha256', 'dataset_sha256', 'model_id', 'model_revision',
              'tokenizer_id', 'tokenizer_revision', 'dtype', 'quantization', 'device_map', 'git_commit'):
        if saved.get(k) != current.get(k):
            raise ValueError(f'provenance mismatch: {k}')


def history_signatures(rows):
    return set(map(concrete_history_signature, rows))


def disjoint(rows, prior_paths):
    current, seen = history_signatures(rows), set()
    for path in prior_paths:
        earlier = history_signatures(read_jsonl(path))
        if current & earlier:
            raise ValueError(f'concrete history overlap with {path}')
        # Historical experiments may overlap each other; only new stages must be fresh.
        seen |= earlier
    return sorted(current)


def validate_dataset(rows, stage):
    audit_behavior_dataset(rows, 'controls_counterbalanced')
    if stage not in COUNTS:
        raise ValueError('unknown dataset stage')
    if len({r['history_id'] for r in rows}) != COUNTS[stage]:
        raise ValueError('wrong stage history count')
    if len(history_signatures(rows)) != COUNTS[stage]:
        raise ValueError('duplicate concrete histories')
    if any(r.get('cross_model_stage') != stage or r.get('cross_model_protocol') != VERSION or
           r.get('prompt_variant') != 'nora_v1' or r.get('prompt_family') != 'natural_entity_attribute_v1' or
           r['candidate_values'] != VALUES or r['seed'] != SEEDS[stage] for r in rows):
        raise ValueError('dataset differs from fixed stage contract')


def evaluate(rows, scores):
    """Recompute every semantic diagnostic from likelihoods; ties fail rank one."""
    expected = {r['example_id']: r for r in rows}
    actual = {r['example_id']: r for r in scores}
    if len(expected) != len(rows) or len(actual) != len(scores) or expected.keys() != actual.keys():
        raise ValueError('dataset/score ID mismatch or duplicate')
    cells, unique, condition_prompts = {}, {}, {}
    for row in rows:
        s = actual[row['example_id']]
        if any(s.get(k) != v for k, v in row.items()):
            raise ValueError('score metadata mismatch')
        if s.get('score_kind') != 'competence_only' or 'matched_edit_effect' in s:
            raise ValueError('gate requires competence-only scores')
        masses = s['semantic_log_mass']
        if set(masses) != set(VALUES) or not all(math.isfinite(v) for v in masses.values()):
            raise ValueError('missing/nonfinite semantic likelihoods')
        target = masses[row['answer']]
        rank = 1 + sum(v >= target for k, v in masses.items() if k != row['answer'])
        margin = target - masses[row['stale_value']] if row['stale_value'] else None
        diagnostic = (int(rank == 1), rank, margin, s['full_vocab_next_token_accuracy'], s.get('exact_token_diagnostic_defined', True))
        signature = (row['prompt_variant'], s['prompt'])
        if signature in unique and unique[signature] != diagnostic:
            raise ValueError('inconsistent duplicate prompt')
        unique[signature] = diagnostic
        condition_prompts.setdefault(row['condition'], {})[signature] = (diagnostic, row['answer'],
                                                                         s.get('generated_first_token'), s.get('full_vocab_rank'))
        if row['condition'] == 'irrelevant':
            continue
        key = (row['prompt_variant'], row['condition'], row['query'], row['orientation'],
               row['edited_variable'], row['edit_status'], row['pair_direction'],
               row.get('unassigned_slot_order') or 'none')
        cells.setdefault(key, {})[signature] = diagnostic
    if set(cells) != expected_cells(['nora_v1']):
        raise ValueError('incomplete expected 64-cell set')
    summary, failures = {}, []
    for key, prompts in sorted(cells.items()):
        ds = list(prompts.values()); margins = [d[2] for d in ds if d[2] is not None]
        metrics = {'n': len(ds), 'semantic_accuracy': sum(d[0] for d in ds)/len(ds),
                   'mean_semantic_rank': sum(d[1] for d in ds)/len(ds),
                   'mean_current_minus_stale': sum(margins)/len(margins) if margins else None,
                   'exact_token_accuracy_diagnostic': sum(d[3] for d in ds if d[4])/sum(d[4] for d in ds) if any(d[4] for d in ds) else None,
                   'exact_token_diagnostic_defined_n': sum(d[4] for d in ds)}
        name = '|'.join(map(str, key)); summary[name] = metrics
        reasons = []
        if metrics['semantic_accuracy'] < GATE['semantic_accuracy_min']: reasons.append('semantic_accuracy')
        if metrics['mean_semantic_rank'] > GATE['mean_semantic_rank_max']: reasons.append('semantic_rank')
        if margins and metrics['mean_current_minus_stale'] <= 0: reasons.append('current_minus_stale')
        if reasons: failures.append({'cell': name, 'reasons': reasons})
    condition_summary = {}
    from collections import Counter
    for condition, prompts in condition_prompts.items():
        entries = list(prompts.values()); ds = [e[0] for e in entries]
        wrong = [e for e in entries if e[0][4] and not e[0][3]]
        condition_summary[condition] = {
            'unique_prompts':len(entries), 'semantic_accuracy':sum(d[0] for d in ds)/len(ds),
            'mean_semantic_rank':sum(d[1] for d in ds)/len(ds),
            'exact_token_defined_n':sum(d[4] for d in ds),
            'exact_token_accuracy':sum(d[3] for d in ds if d[4])/sum(d[4] for d in ds) if any(d[4] for d in ds) else None,
            'case_only_next_token_misses_diagnostic':sum(str(e[2]).strip().lower()==e[1] for e in wrong),
            'first_token_exact_surface_parse_rate_diagnostic':sum(
                e[2] in {prefix+v for v in VALUES for prefix in ('',' ') } |
                        {prefix+v.title() for v in VALUES for prefix in ('',' ')} for e in entries)/len(entries),
            'common_raw_greedy_tokens':dict(Counter(e[2] for e in entries).most_common(10)),
            'normalization_note':'case classification is diagnostic only; no outputs or probabilities normalized'}
    return {'pass': not failures, 'condition_summary':condition_summary, 'cells': summary, 'failed_cells': failures,
            'expected_cell_count': 64, 'unique_prompts': len(unique)}
