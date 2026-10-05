#!/usr/bin/env python3
"""Run resumable model scoring for the separate relational follow-ups."""
import argparse
import importlib.metadata
import json
import random
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.cross_model.followups import (VERSION, analyze_marker, complete_answer_outcome,
    analyze_distance, analyze_harder, generate_harder, generate_marker, choose_difficulty,
    validate_distance_dataset, validate_harder_dataset, validate_marker_dataset)

from src.cross_model.chains import analyze_chain, validate_chain_dataset
from src.cross_model.update_control import analyze_update, validate_update_dataset

VALIDATORS = {'marker': validate_marker_dataset, 'harder': validate_harder_dataset,
              'distance': validate_distance_dataset, 'chain': validate_chain_dataset,
              'update': validate_update_dataset}
ANALYSES = {'marker': analyze_marker, 'harder': analyze_harder, 'distance': analyze_distance,
            'chain': analyze_chain, 'update': analyze_update}
VALID_STAGES = {'marker': {'pilot', 'confirmatory'}, 'harder': {'pilot', 'development', 'test'},
                'distance': {'pilot', 'confirmatory'}, 'chain': {'development', 'confirmatory'},
                'update': {'pilot', 'confirmatory'}}
from src.cross_model.protocol import VALUES, digest, sealed, write_new, read_sealed
from src.cross_model.progress import progress
from src.cross_model.scoring import score_prompt
from src.cross_model.tokens import continuations, encode
from src.data.io import read_jsonl, sha256_file
from src.data.supersession_behavior import _answer_prefix
from src.utils import load_config, provenance


def _code_hash():
    paths = ('src/cross_model/followups.py', 'scripts/followups.py', 'scripts/run_followups.py',
             'src/cross_model/scoring.py', 'src/cross_model/tokens.py',
             'src/models/loader.py', 'src/data/supersession_behavior.py',
             'src/cross_model/runtime.py', 'src/cross_model/protocol.py',
             'src/data/io.py', 'src/utils.py', 'pyproject.toml', 'uv.lock')
    return digest({p: sha256_file(p) for p in paths})


def _runtime_fingerprint(config, config_path, model, tokenizer, torch):
    packages = {}
    for package in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'tokenizers'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    quantization = getattr(model.config, 'quantization_config', None)
    if hasattr(quantization, 'to_dict'):
        quantization = quantization.to_dict()
    elif not isinstance(quantization, (dict, list, str, int, float, bool, type(None))):
        quantization = repr(quantization)
    quantization = json.loads(json.dumps(quantization, default=str))
    tokenizer_kwargs = json.loads(json.dumps(getattr(tokenizer, 'init_kwargs', {}), default=str))
    model_config = json.loads(json.dumps(model.config.to_dict(), default=str))
    return {
        'python_version': platform.python_version(),
        'uv_lock_sha256': sha256_file('uv.lock'),
        'configuration_sha256': sha256_file(config_path),
        'configuration_digest': digest(config),
        'model_config_digest': digest(model_config),
        'model_class': f'{model.__class__.__module__}.{model.__class__.__qualname__}',
        'configured_model': config.get('model', {}),
        'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
        'parameter_devices': sorted({str(p.device) for p in model.parameters()}),
        'model_quantization_config': quantization,
        'model_attention_implementation': getattr(model.config, '_attn_implementation', None),
        'tokenizer_class': f'{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}',
        'tokenizer_name_or_path': getattr(tokenizer, 'name_or_path', None),
        'tokenizer_init_kwargs_digest': digest(tokenizer_kwargs),
        'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()),
        'chat_template_sha256': digest(tokenizer.chat_template),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_capability': list(torch.cuda.get_device_capability(0)),
        'packages': packages,
    }


def _prompt(row, tokenizer):
    return _answer_prefix(row['prompt'], tokenizer, True)


def _generate(model, tokenizer, prompt, max_new_tokens):
    import torch
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    device = model.get_input_embeddings().weight.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, num_beams=1,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    generated = output[0, encoded['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.numel())


def _load_model(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('follow-up inference requires CUDA; no CPU fallback')
    from src.cross_model.runtime import load_pinned_model
    model, tokenizer = load_pinned_model(config)
    model.eval()
    return model, tokenizer


def validate_test_freeze(frozen):
    if (frozen.get('protocol') != VERSION
            or frozen.get('experiment') != 'harder'
            or frozen.get('stage') != 'frozen_test_protocol'
            or not frozen.get('selection_gate_passed')):
        raise ValueError('harder test requires a qualifying sealed protocol with development accuracy in [0.65,0.90]')


def validate_test_runtime(frozen, model_config, runtime_fingerprint):
    if model_config['id'] == frozen['model_id']:
        development_runtime = frozen['development_levels'][str(frozen['selected_n_distractors'])]['runtime_fingerprint']
        if (model_config['revision'] != frozen['model_revision']
                or runtime_fingerprint != development_runtime):
            raise ValueError('Qwen test runtime differs from frozen development runtime')


def dataset_lineage(dataset, rows, report_path=None):
    """Authenticate the generation report and retain its explicit exclusions."""
    path = Path(report_path) if report_path else Path(dataset).with_name(Path(dataset).stem + '_report.json')
    if not path.exists():
        if rows[0]['stage'] != 'pilot':
            raise ValueError('nonpilot scoring requires its generation report; use --dataset-report for a custom path')
        return {'exclusions': [], 'prior_datasets': [], 'dataset_report_sha256': None}
    report = json.loads(path.read_text())
    if report.get('dataset_sha256') != sha256_file(dataset) or report.get('stage') != rows[0]['stage']:
        raise ValueError('generation report does not authenticate this dataset')
    exclusions = set(report.get('exclusions', []))
    ledger = set()
    for prior in report.get('prior_datasets', []):
        prior_path = Path(prior['path'])
        # Stored absolute paths may refer to the original GPU host checkout.
        if not prior_path.exists():
            prior_path = Path(dataset).parent / prior_path.name
        if not prior_path.exists() or sha256_file(prior_path) != prior['sha256']:
            raise ValueError('prior dataset exclusion ledger is missing or changed')
        for row in read_jsonl(prior_path):
            ledger.update((row['history_signature'], row['target_history_signature']))
    if ledger != exclusions:
        # Test exclusions also include the sealed development history set.
        if rows[0]['stage'] != 'test' or not ledger <= exclusions:
            raise ValueError('generation exclusions differ from prior datasets')
    first_stage_of_new_design = ((rows[0].get('condition') == 'superseded_chain' and rows[0]['stage'] == 'development')
                                 or rows[0]['history_id'].startswith('relational_followups_update_v1:'))
    if rows[0]['stage'] != 'pilot' and not ledger and not first_stage_of_new_design:
        raise ValueError('nonpilot dataset has no explicit earlier-history exclusions')
    if any(r['history_signature'] in exclusions or r['target_history_signature'] in exclusions for r in rows):
        raise ValueError('dataset overlaps its explicit earlier-history exclusions')
    return {'exclusions': sorted(exclusions), 'prior_datasets': report.get('prior_datasets', []),
            'dataset_report_sha256': sha256_file(path)}


def run(args):
    config = load_config(args.config)
    rows = read_jsonl(args.dataset)
    if not rows:
        raise ValueError('empty dataset')
    stages = {r.get('stage') for r in rows}
    if len(stages) != 1:
        raise ValueError('dataset contains mixed or missing stage labels')
    stage = next(iter(stages))
    if stage not in VALID_STAGES[args.experiment]:
        raise ValueError(f'{args.experiment} does not define stage {stage!r}')
    if args.experiment == 'harder' and args.mode != 'both':
        raise ValueError('harder experiment requires --mode both (answers and candidate scores)')
    if args.experiment in ('marker', 'distance', 'update', 'chain') and args.mode not in ('candidate', 'both'):
        raise ValueError(f'{args.experiment} experiment requires candidate scores')
    VALIDATORS[args.experiment](rows)
    if args.experiment == 'harder' and stage == 'test':
        if not args.freeze:
            raise ValueError('scoring test histories requires --freeze')
        frozen = read_sealed(args.freeze)
        validate_test_freeze(frozen)
        if (frozen.get('prompt_code_sha256') != _code_hash()
                or frozen.get('selected_n_distractors') != rows[0].get('n_distractors')
                or frozen.get('max_new_tokens') != args.max_new_tokens
                or frozen.get('scoring_seed') != args.seed
                or len({r['history_id'] for r in rows}) != frozen.get('test_histories')
                or {r['seed'] for r in rows} != {frozen.get('test_dataset_seed') + 1_000_003}):
            raise ValueError('test dataset/scoring settings differ from frozen development protocol')
        test_targets = {r['target_history_signature'] for r in rows}
        if test_targets & set(frozen.get('excluded_history_signatures', [])):
            raise ValueError('test target histories overlap frozen development histories')
    lineage = dataset_lineage(args.dataset, rows, getattr(args, 'dataset_report', None))
    out = Path(args.output)
    run_path, prov_path = Path(str(out) + '.run.json'), Path(str(out) + '.provenance.json')
    if args.resume and not run_path.exists():
        raise ValueError('--resume requires matching .run.json')
    if not args.resume and (out.exists() or run_path.exists() or prov_path.exists()):
        raise FileExistsError('output bundle exists; choose a fresh path or use --resume')
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    load_start = time.perf_counter()
    model, tokenizer = _load_model(config)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError('pinned tokenizer chat template required')
    dataset_hash, code_hash = sha256_file(args.dataset), _code_hash()
    freeze_hash = sha256_file(args.freeze) if args.freeze else None
    runtime_fingerprint = _runtime_fingerprint(config, args.config, model, tokenizer, torch)
    if args.experiment == 'harder' and stage == 'test':
        validate_test_runtime(frozen, config['model'], runtime_fingerprint)
    fingerprint = {'protocol': VERSION, 'experiment': args.experiment,
        'stage': stage, 'mode': args.mode, 'model_id': config['model']['id'],
        'model_revision': config['model']['revision'],
        'tokenizer_revision': config['model'].get('tokenizer_revision', config['model']['revision']),
        'dataset_sha256': dataset_hash, 'code_sha256': code_hash,
        'seed': args.seed, 'max_new_tokens': args.max_new_tokens,
        'protocol_freeze_sha256': freeze_hash,
        'decoding': 'greedy; do_sample=false; num_beams=1',
        'parser': 'trim whitespace and edge punctuation; exact case-insensitive candidate match',
        'runtime_fingerprint': runtime_fingerprint}
    fingerprint['dataset_report_sha256'] = lineage['dataset_report_sha256']
    if args.resume:
        saved = json.loads(run_path.read_text())
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('resume fingerprint mismatch: configuration, model runtime, tokenizer, packages, data or code changed')
        if prov_path.exists():
            raise ValueError('run is already complete')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_new(run_path, {'fingerprint': fingerprint, 'status': 'started',
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'runtime_fingerprint': runtime_fingerprint})
    device = model.get_input_embeddings().weight.device
    scored = read_jsonl(out) if out.exists() else []
    expected = {r['example_id']: r for r in rows}
    prompts = {r['example_id']: _prompt(r, tokenizer) for r in rows}
    prompt_answers = {}
    for row in rows:
        prompt = prompts[row['example_id']]
        if prompt in prompt_answers and prompt_answers[prompt] != row['answer']:
            raise ValueError('same rendered prompt has conflicting correct answers')
        prompt_answers[prompt] = row['answer']
    done = {}
    cache = {}
    for saved in scored:
        eid = saved.get('example_id')
        if eid not in expected or eid in done or any(saved.get(k) != v for k, v in expected[eid].items()):
            raise ValueError('checkpoint record does not match dataset')
        done[eid] = saved
        cache[saved['rendered_prompt']] = {k: saved[k] for k in ('semantic_log_mass', 'surface_likelihoods',
            'generated_answer', 'parsed_answer', 'answer_category', 'generation_token_count') if k in saved}
    runtime_start = time.perf_counter()
    resumed_unique_prompts = len(cache)
    with out.open('a', encoding='utf8') as handle:
        for row in progress(rows, desc=f'Scoring {args.experiment}', unit='member'):
            if row['example_id'] in done:
                continue
            prompt = prompts[row['example_id']]
            core = cache.get(prompt)
            if core is None:
                candidates = row.get('candidate_values', VALUES)
                events = continuations(tokenizer, prompt, candidates)
                if set(events) != set(candidates):
                    raise ValueError('candidate continuation map must cover the row candidate vocabulary')
                masses, surfaces, _, _ = score_prompt(model, tokenizer, prompt, events)
                core = {'semantic_log_mass': masses, 'surface_likelihoods': surfaces}
                if args.mode == 'both':
                    answer, token_count = _generate(model, tokenizer, prompt, args.max_new_tokens)
                    outcome = complete_answer_outcome(answer, row['answer'], row.get('stale_value'),
                                                      row.get('candidate_values', VALUES))
                    core.update(generated_answer=answer, parsed_answer=outcome['parsed_answer'],
                                answer_category=outcome['category'], generation_token_count=token_count)
                cache[prompt] = core
            stale = row.get('stale_value') or row.get('historical_values', {}).get(row.get('query_entity'))
            margin = core['semantic_log_mass'][row['answer']] - core['semantic_log_mass'][stale] if stale else None
            record = {**row, **core, 'rendered_prompt': prompt,
                      'current_minus_historical_log_mass': margin}
            parsed = core.get('parsed_answer')
            record['query_source_response'] = parsed == row.get('source_value_for_query', row.get('source_value'))
            record['query_donor_response'] = parsed == row.get('replacement_value_for_query', row.get('replacement_value'))
            record['edited_source_response'] = parsed == row.get('source_value')
            record['edited_donor_response'] = parsed == row.get('replacement_value')
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            done[row['example_id']] = record
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - runtime_start
    if set(done) != set(expected):
        raise ValueError('incomplete scored dataset')
    config_prov = provenance(config, args.dataset)
    runtime = {'python': config_prov['python'], 'packages': config_prov['packages'],
        'cuda_version': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(0),
        'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
        'tokenizer_vocab_sha256': digest(tokenizer.get_vocab()),
        'chat_template_sha256': digest(tokenizer.chat_template),
        'model_load_seconds': model_load_seconds,
        'elapsed_seconds': elapsed, 'unique_prompts': len(cache),
        'runtime_scope': 'current invocation; earlier interrupted invocations excluded',
        'resumed_unique_prompts': resumed_unique_prompts,
        'new_unique_prompts': len(cache) - resumed_unique_prompts,
        'seconds_per_unique_prompt': elapsed / (len(cache) - resumed_unique_prompts)
            if len(cache) > resumed_unique_prompts else None}
    report = {'fingerprint': fingerprint, 'dataset_path': str(Path(args.dataset).resolve()),
        'dataset_sha256': dataset_hash, 'scores_sha256': sha256_file(out),
        'model_config_sha256': digest(json.loads(model.config.to_json_string())),
        'runtime': runtime, 'n_members': len(rows), 'n_unique_prompts': len(cache),
        'n_histories': len({r['history_id'] for r in rows}),
        'n_distractors': rows[0].get('n_distractors'),
        'history_signatures': sorted({r['history_signature'] for r in rows}),
        'target_history_signatures': sorted({r['target_history_signature'] for r in rows}),
        **lineage,
        'status': 'complete'}
    write_new(prov_path, sealed(report))
    run_path.write_text(json.dumps({'fingerprint': fingerprint, 'status': 'complete',
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scores_sha256': report['scores_sha256']}, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def summarize_harder(scores):
    by = defaultdict(list)
    for row in scores:
        by[(row['condition'], row['edited'])].append(row)
    return {f'{condition}_edited_{edited}': {
        'n_rows': len(rows), 'accuracy': float(np.mean([r['answer_category'] == 'correct' for r in rows])),
        'stale_frequency': float(np.mean([r['answer_category'] == 'stale' for r in rows])),
        'mean_current_minus_historical_log_mass': float(np.mean([r['current_minus_historical_log_mass'] for r in rows if r['current_minus_historical_log_mass'] is not None]))}
        for (condition, edited), rows in sorted(by.items())}


def analyze(args):
    rows, scores = read_jsonl(args.dataset), read_jsonl(args.scores)
    stages = {r.get('stage') for r in rows}
    if len(stages) != 1 or stages != {args.stage}:
        raise ValueError(f'analysis stage {args.stage!r} does not match dataset stage(s) {sorted(stages)}')
    if args.stage not in VALID_STAGES[args.experiment]:
        raise ValueError(f'{args.experiment} does not define stage {args.stage!r}')
    score_prov_path = Path(args.scores + '.provenance.json')
    if not score_prov_path.exists():
        raise ValueError('analysis requires the completed score provenance sidecar')
    score_prov = read_sealed(score_prov_path)
    expected_dataset_hash, expected_scores_hash = sha256_file(args.dataset), sha256_file(args.scores)
    fingerprint = score_prov.get('fingerprint', {})
    if (score_prov.get('status') != 'complete'
            or score_prov.get('dataset_sha256') != expected_dataset_hash
            or score_prov.get('scores_sha256') != expected_scores_hash
            or fingerprint.get('experiment') != args.experiment
            or fingerprint.get('dataset_sha256') != expected_dataset_hash):
        raise ValueError('score provenance does not authenticate this dataset and score file')
    result = ANALYSES[args.experiment](rows, scores)
    result.update(protocol=VERSION, experiment=args.experiment, stage=args.stage,
                  dataset_sha256=sha256_file(args.dataset), scores_sha256=sha256_file(args.scores),
                  code_sha256=_code_hash(), dataset_seeds=sorted({r['seed'] for r in rows}),
                  n_distractors=rows[0].get('n_distractors'),
                  history_signatures=sorted({r['history_signature'] for r in rows}),
                  target_history_signatures=sorted({r['target_history_signature'] for r in rows}),
                  exclusions=score_prov.get('exclusions', []),
                  prior_datasets=score_prov.get('prior_datasets', []),
                  dataset_report_sha256=score_prov.get('dataset_report_sha256'))
    result['inference_provenance_sha256'] = sha256_file(score_prov_path)
    result['inference_provenance'] = score_prov
    write_new(args.output, sealed(result))
    print(json.dumps({'experiment': args.experiment, 'stage': args.stage,
        'output': args.output, 'status': 'analysis_complete'}, indent=2))


def select_difficulty(args):
    reports = {}
    for level, path in zip((2, 4, 6), args.development_reports):
        report = read_sealed(path)
        if report.get('protocol') != VERSION or report.get('experiment') != 'harder' or report.get('stage') != 'development':
            raise ValueError('difficulty selection requires sealed harder-task development analyses')
        fingerprint = report.get('inference_provenance', {}).get('fingerprint', {})
        if fingerprint.get('experiment') != 'harder' or fingerprint.get('code_sha256') != _code_hash():
            raise ValueError('development report lacks authenticated inference provenance')
        if report.get('code_sha256') != _code_hash():
            raise ValueError('development analysis was produced by a different code version')
        reports[str(level)] = {'accuracy': report['overall_complete_answer_accuracy'],
            'analysis_path': str(Path(path).resolve()), 'analysis_sha256': sha256_file(path),
            'dataset_sha256': report['dataset_sha256'], 'scores_sha256': report['scores_sha256'],
            'dataset_seeds': report['dataset_seeds'],
            'model_id': fingerprint['model_id'], 'model_revision': fingerprint['model_revision'],
            'configuration_sha256': fingerprint['runtime_fingerprint']['configuration_sha256'],
            'runtime_fingerprint_sha256': digest(fingerprint['runtime_fingerprint']),
            'runtime_fingerprint': fingerprint['runtime_fingerprint'],
            'n_distractors': report.get('n_distractors'),
            'history_signatures': report.get('history_signatures', []),
            'target_history_signatures': report.get('target_history_signatures', []),
            'n_histories': len(report.get('target_history_signatures', [])),
            'max_new_tokens': fingerprint['max_new_tokens'], 'scoring_seed': fingerprint['seed']}
        if not np.isfinite(reports[str(level)]['accuracy']) or not 0 <= reports[str(level)]['accuracy'] <= 1:
            raise ValueError('development accuracy must be finite and between zero and one')
    models = {(x['model_id'], x['model_revision']) for x in reports.values()}
    if len(models) != 1:
        raise ValueError('all three development levels must use the same model and immutable revision')
    if next(iter(models))[0] != 'Qwen/Qwen3-8B':
        raise ValueError('difficulty selection requires Qwen development reports')
    max_tokens = {x['max_new_tokens'] for x in reports.values()}
    scoring_seeds = {x['scoring_seed'] for x in reports.values()}
    if len(max_tokens) != 1 or len(scoring_seeds) != 1:
        raise ValueError('development levels must use identical answer-generation settings')
    if len({v['runtime_fingerprint_sha256'] for v in reports.values()}) != 1:
        raise ValueError('development levels used incompatible configurations or runtimes')
    if len({tuple(v['dataset_seeds']) for v in reports.values()}) != 1:
        raise ValueError('development levels must use the same dataset seed')
    if any(v['n_distractors'] != int(level) for level, v in reports.items()):
        raise ValueError('development report difficulty does not match its declared level')
    if len({v['n_histories'] for v in reports.values()}) != 1:
        raise ValueError('development levels must use the same history count')
    if any(v['n_histories'] != 12 for v in reports.values()):
        raise ValueError('difficulty selection requires 12 histories per development level')
    signature_sets = [set(v['target_history_signatures']) for v in reports.values()]
    if not signature_sets[0] or any(s != signature_sets[0] for s in signature_sets[1:]):
        raise ValueError('development difficulty levels must share exactly matched physical histories')
    if all(v['accuracy'] > .90 for v in reports.values()):
        raise ValueError('amended stopping rule: all Qwen development levels exceed 90%; stop without selecting a fallback level or sealing a test protocol')
    selected = choose_difficulty(reports)
    frozen = {'protocol': VERSION, 'experiment': 'harder', 'stage': 'frozen_test_protocol',
        'selected_n_distractors': selected,
        'selection_rule': 'hardest development accuracy in [0.65,0.90]; otherwise nearest 0.775, ties smaller level',
        'stopping_rule_amendment': 'post-pilot, pre-development: stop if all Qwen levels exceed 0.90; test requires a level in [0.65,0.90]',
        'development_levels': reports, 'model_id': next(iter(models))[0],
        'model_revision': next(iter(models))[1], 'difficulty_levels': [2,4,6],
        'prompt_code_sha256': _code_hash(), 'candidate_values': VALUES,
        'prompt_templates': {'historical_assignment': 'Previously, {entity}\'s {attribute} was {value}.',
            'entity_mention': '{entity} mentioned {value} in an unrelated note.',
            'unassigned': 'The unassigned {attribute} value was {value}.',
            'current_assignment': 'Currently, {entity}\'s {attribute} is {value}.',
        'distractor': "Previously, {entity}'s badge was {old}.\nCurrently, {entity}'s badge is {current}.",
            'query': 'What is {entity}\'s current {attribute}?',
            'instruction': 'Respond with only the value, with no explanation.'},
        'decoding': 'greedy; do_sample=false; num_beams=1',
        'max_new_tokens': next(iter(max_tokens)),
        'parser': 'trim whitespace and edge punctuation; exact case-insensitive candidate match',
        'metrics': ['complete_answer_accuracy', 'stale_answer_frequency', 'paired_transitions',
                    'current_minus_historical_candidate_log_mass', 'secondary_R'],
        'bootstrap': {'unit': 'history', 'draws': 2000, 'seed': 73021},
        'test_dataset_seed': 20261004, 'exclusions': [],
        'test_histories': 24,
        'development_histories_per_level': next(iter({v['n_histories'] for v in reports.values()})),
        'excluded_history_signatures': sorted(set.union(*signature_sets))}
    eligible = [v['accuracy'] for v in reports.values() if .65 <= v['accuracy'] <= .90]
    frozen['selection_gate_passed'] = bool(eligible)
    frozen['failed_gates'] = [] if eligible else ['no_development_level_accuracy_in_0.65_to_0.90; nearest_0.775_fallback_used']
    frozen['scoring_seed'] = next(iter(scoring_seeds))
    write_new(args.output, sealed(frozen))
    print(json.dumps({'selected_n_distractors': selected, 'output': args.output,
        'development_accuracy': {k:v['accuracy'] for k,v in reports.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    score = sub.add_parser('score')
    score.add_argument('--experiment', choices=('marker', 'harder', 'distance', 'chain', 'update'), required=True)
    score.add_argument('--mode', choices=('candidate', 'both'), default='both')
    score.add_argument('--config', required=True); score.add_argument('--dataset', required=True)
    score.add_argument('--dataset-report', help='generation report; defaults to DATASET_STEM_report.json')
    score.add_argument('--output', required=True); score.add_argument('--seed', type=int, default=20261006)
    score.add_argument('--max-new-tokens', type=int, default=32); score.add_argument('--resume', action='store_true')
    score.add_argument('--freeze')
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--experiment', choices=('marker', 'harder', 'distance', 'chain', 'update'), required=True)
    analysis.add_argument('--stage', choices=('pilot','development','test','confirmatory'), required=True)
    analysis.add_argument('--dataset', required=True); analysis.add_argument('--scores', required=True)
    analysis.add_argument('--output', required=True)
    select = sub.add_parser('select-hard-difficulty')
    select.add_argument('--development-reports', nargs=3, required=True, metavar=('N2_REPORT','N4_REPORT','N6_REPORT'))
    select.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'score': run(args)
        elif args.command == 'analyze': analyze(args)
        else: select_difficulty(args)
    except Exception as exc:
        if args.command == 'score':
            failure = Path(str(args.output) + '.failures.jsonl')
            failure.parent.mkdir(parents=True, exist_ok=True)
            with failure.open('a', encoding='utf8') as handle:
                handle.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'experiment': args.experiment, 'model_config': args.config,
                    'error_type': type(exc).__name__, 'error': str(exc),
                    'dataset_sha256': sha256_file(args.dataset) if Path(args.dataset).exists() else None,
                    'code_sha256': _code_hash()}, sort_keys=True) + '\n')
        raise


if __name__ == '__main__':
    main()
