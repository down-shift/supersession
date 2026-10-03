#!/usr/bin/env python3
"""Separately frozen relational stages and read-only archived v1 reanalysis."""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from scripts.cross_model import fresh_bundle, tokenizer, write_dataset
from src.cross_model import robustness_protocol as design
from src.cross_model import robustness_v2 as data
from src.cross_model.progress import configure_logging, progress
from src.cross_model.protocol import digest, read_sealed, sealed, write_new
from src.cross_model.tokens import check_tokenizer, surface_geometry_audit, validate as validate_tokens
from src.cross_model.workflow import (dataset_info, gate_report, local_artifact_path,
                                      score_info, verify_confirmation, verify_gate)
from src.data.io import read_jsonl, sha256_file
from src.utils import load_config, provenance

logger = logging.getLogger('relational_robustness')


def sample_pairs(rows):
    """Every condition/order/edit/query cell for the first history of each orientation."""
    first = {o: min(r['history_index'] for r in rows if r['orientation'] == o) for o in (0, 1)}
    pairs = {}
    for row in rows:
        if row['history_index'] != first[row['orientation']]:
            continue
        value = pairs.setdefault(row['pair_id'], {'condition': row['condition'], 'orientation': row['orientation'],
              'historical_order': row['historical_entity_order'], 'current_order': row['current_entity_order'],
              'query_entity': row['query_entity'], 'edited_entity': row['edited_entity'],
              'source_value': row['source_value'], 'replacement_value': row['replacement_value']})
        value['baseline' if row['pair_direction'] == 0 else 'edited'] = {
            'prompt': data.render(row, None, False), 'answer': row['answer']}
    return dict(sorted(pairs.items()))


def prompt_audit(stage, prior_paths):
    histories, rows = data.generate(stage, prior_paths)
    data.validate(histories, rows, stage)
    return {'protocol': data.VERSION, 'design_revision': data.DESIGN_REVISION,
            'contract': design.CONTRACT, 'stage': stage, 'code_sha256': design.code_hash(),
            'n_histories': len(histories), 'n_members': len(rows), 'n_pairs': len(rows)//2,
            'concrete_history_signatures': data.disjoint(rows, prior_paths),
            'prior_datasets': [{'path': str(Path(p).resolve()), 'sha256': sha256_file(p)} for p in prior_paths],
            'sample_pairs': sample_pairs(rows),
            'status': 'tokenizer-free design audit; not ready for inference',
            'provenance': provenance({}, None)}


def token_audit(tok, rows):
    if not getattr(tok, 'chat_template', None):
        raise ValueError('pinned chat template required; no raw-prompt fallback for tokenizer/scoring stages')
    result = validate_tokens(tok, rows, renderer=data.render, positions=data.semantic_positions)
    geometry, links = {}, {}
    for row in progress(rows, desc='Recording value/entity token offsets', unit='row'):
        prompt_hash = digest(data.render(row, tok, True))
        if prompt_hash not in geometry:
            entry = data.span_audit(row, tok)
            entry.pop('example_id')
            geometry[prompt_hash] = entry
        links[row['example_id']] = prompt_hash
    result['token_span_geometry'] = {'by_prompt_sha256': geometry, 'example_to_prompt_sha256': links,
                                     'index_convention': 'zero-based token offsets; end exclusive; all value-span tokens included'}
    return result


def load_candidate(args, config):
    if not args.candidates:
        raise ValueError('--candidates is required for this stage')
    candidate = read_sealed(args.candidates)
    if candidate.get('contract') != design.CONTRACT:
        raise ValueError('relational candidate contract mismatch')
    frozen_path = local_artifact_path(candidate['freeze_path'])
    design.verify_freeze(frozen_path)
    if sha256_file(frozen_path) != candidate['freeze_sha256']:
        raise ValueError('candidate map freeze binding changed')
    design.check_manifest(candidate['provenance'], config, args.config)
    if candidate['surface_geometry_audit'] != surface_geometry_audit(candidate['events']):
        raise ValueError('mandatory surface geometry missing or inconsistent')
    return candidate


def prior_paths(args):
    return list(dict.fromkeys(design.discover_prior_datasets(args.stage) + args.prior_dataset))


def claim_stage(config, stage, output):
    key = digest({'model': config['model']['id'], 'revision': config['model']['revision'],
                  'contract': design.CONTRACT})[:16]
    path = Path('outputs/cross_model_relational_v2/stage_claims') / key / f'{stage}.json'
    write_new(path, sealed({'stage': stage, 'dataset_path': str(Path(output).resolve()),
                            'model': config['model'], 'contract_sha256': digest(design.CONTRACT)}))


def register_shared_histories(histories, stage):
    path = Path('outputs/cross_model_relational_v2/shared_histories') / f'{data.DESIGN_REVISION}_{stage}.json'
    value = {'stage': stage, 'design_revision': data.DESIGN_REVISION, 'histories': histories,
             'contract_sha256': digest(design.CONTRACT)}
    if path.exists():
        if read_sealed(path) != value:
            raise ValueError('shared histories differ across models')
    else:
        write_new(path, sealed(value))
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def generate_stage(args, config, candidate):
    if not args.stage:
        raise ValueError('--stage is required')
    fresh_bundle(args.output)
    priors = prior_paths(args)
    info = {'stage': args.stage, 'freeze_path': candidate['freeze_path'], 'freeze_sha256': candidate['freeze_sha256']}
    if args.stage == 'frozen_gate':
        if not args.development_report:
            raise ValueError('--development-report is required before generating the fresh gate')
        dev = read_sealed(args.development_report)
        dr, ds, _, _ = score_info(dev['scores_path'], dev['dataset_path'], config, args.config,
                                  args.candidates, 'development', design=design)
        if design.evaluate(dr, ds, gate=False) != dev['evaluation']:
            raise ValueError('development report does not recompute')
        info.update(development_report_path=str(Path(args.development_report).resolve()),
                    development_report_sha256=sha256_file(args.development_report))
        priors.append(str(local_artifact_path(dev['dataset_path'])))
    elif args.stage == 'confirmatory':
        if not args.gate or not args.preflight:
            raise ValueError('--gate and --preflight are required before confirmation generation')
        gate = verify_gate(args.gate, config, args.config, args.candidates, design=design)
        preflight = read_sealed(args.preflight)
        if preflight.get('stage') != 'preflight' or preflight['gate_sha256'] != sha256_file(args.gate):
            raise ValueError('incorrect preflight/gate binding')
        design.check_manifest(preflight['provenance'], config, args.config, args.candidates,
                              local_artifact_path(gate['dataset_path']))
        dev = read_sealed(local_artifact_path(gate['development_report_path']))
        priors.extend([str(local_artifact_path(gate['dataset_path'])), str(local_artifact_path(dev['dataset_path']))])
        info.update(gate_path=str(Path(args.gate).resolve()), gate_sha256=sha256_file(args.gate),
                    preflight_path=str(Path(args.preflight).resolve()), preflight_sha256=sha256_file(args.preflight))
    priors = list(dict.fromkeys(priors))
    histories, rows = data.generate(args.stage, priors)
    tok = tokenizer(config, args.local_files_only)
    check_tokenizer(tok, candidate)
    audit = token_audit(tok, rows)
    if audit['events'] != candidate['events']:
        raise ValueError('actual stage candidate map differs from the frozen map')
    geometry_path = args.output + '.geometry.json'
    if Path(geometry_path).exists():
        raise FileExistsError(geometry_path)
    info.update(history_signatures=data.disjoint(rows, priors),
                prior_datasets=[{'path': str(Path(p).resolve()), 'sha256': sha256_file(p)} for p in priors],
                edit_audit=audit['edit_audit'], exhaustive_slot_audit=audit['exhaustive_slot_audit'],
                tokenizer_sha256=candidate['tokenizer_sha256'], chat_template_sha256=candidate['chat_template_sha256'],
                shared_history_registry=register_shared_histories(histories, args.stage))
    claim_stage(config, args.stage, args.output)
    write_dataset(args.output, rows, info)
    write_new(geometry_path, sealed(audit['token_span_geometry']))
    info.update(geometry_path=str(Path(geometry_path).resolve()), geometry_sha256=sha256_file(geometry_path),
                provenance=design.manifest(config, args.config, args.candidates, args.output))
    write_new(args.output + '.provenance.json', sealed(info))
    print(json.dumps({'stage': args.stage, 'histories': len(histories), 'members': len(rows), 'output': args.output}))


def score_stage(args, config, candidate):
    from src.cross_model.score_checks import checked_scores, strict_rank
    rows, info = dataset_info(args.dataset, config, args.config, args.candidates, design=design)
    if info['stage'] == 'confirmatory':
        verify_confirmation(args.dataset, config, args.config, args.candidates, design=design)
        gate = read_sealed(local_artifact_path(info['gate_path']))
        old = read_sealed(str(local_artifact_path(gate['scores_path'])) + '.provenance.json')['provenance']
        current = design.manifest(config, args.config, args.candidates, args.dataset)
        if any(old[k] != current[k] for k in ('packages', 'python')):
            raise ValueError('confirmation runtime differs from gate runtime')
    if not args.resume:
        fresh_bundle(args.output)
    if Path(args.output + '.provenance.json').exists():
        score_info(args.output, args.dataset, config, args.config, args.candidates, info['stage'], design=design)
        checked_scores(rows, read_jsonl(args.output), require_surfaces=True)
        print('already complete')
        return
    from src.cross_model.runtime import load_pinned_model
    from src.cross_model.scoring import score_row
    from src.data.progress import append_jsonl_record, prepare_jsonl_progress
    model, tok = load_pinned_model(config)
    check_tokenizer(tok, candidate)
    audit = token_audit(tok, rows)
    if audit['events'] != candidate['events']:
        raise ValueError('loaded tokenizer candidate map changed')
    prov = design.manifest(config, args.config, args.candidates, args.dataset)
    import torch
    prov.update(model_config_sha256=digest(json.loads(model.config.to_json_string())),
                tokenizer_sha256=candidate['tokenizer_sha256'], chat_template_sha256=candidate['chat_template_sha256'],
                resolved_device_map={k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
                runtime={'cuda_version': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(0),
                         'allocated_bytes': torch.cuda.memory_allocated(0), 'reserved_bytes': torch.cuda.memory_reserved(0),
                         'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()})},
                freeze_sha256=candidate['freeze_sha256'], prompt_edit_audit=audit['edit_audit'])
    fingerprint = {k: v for k, v in prov.items() if k != 'timestamp_utc'}
    fingerprint['runtime'] = {k: v for k, v in prov['runtime'].items() if k not in ('allocated_bytes', 'reserved_bytes')}
    completed = prepare_jsonl_progress(args.output, args.dataset, args.candidates, fingerprint, rows, resume=args.resume)
    expected = {r['example_id']: r for r in rows}
    cache = {}
    for saved in read_jsonl(args.output) if Path(args.output).exists() else []:
        row = expected[saved['example_id']]
        if any(saved.get(k) != v for k, v in row.items()) or saved['prompt'] != data.render(row, tok, True):
            raise ValueError('resume prompt or semantic metadata differs')
        if strict_rank(saved['semantic_log_mass'], row['answer']) != saved['semantic_rank']:
            raise ValueError('resume saved rank differs from masses')
        cache[saved['prompt']] = saved
    for row in progress(rows, desc=f'Scoring {info["stage"]}', unit='member'):
        if row['example_id'] in completed:
            continue
        prompt = data.render(row, tok, True)
        if prompt not in cache:
            cache[prompt] = score_row(model, tok, row, candidate, info['stage'] != 'confirmatory', renderer=data.render)
        if cache[prompt]['answer'] != row['answer']:
            raise ValueError('same prompt unexpectedly has two semantic answers')
        append_jsonl_record(args.output, {**cache[prompt], **row})
    checked_scores(rows, read_jsonl(args.output), require_surfaces=True)
    write_new(args.output + '.provenance.json', sealed({'stage': info['stage'],
              'scores_sha256': sha256_file(args.output), 'provenance': prov}))
    print(json.dumps({'stage': info['stage'], 'members': len(rows), 'causal_effects_computed': False}))


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preview', 'audit', 'freeze', 'validate', 'generate', 'score',
                                            'analyze', 'preflight', 'plan', 'reanalyze'])
    parser.add_argument('--stage', choices=list(data.COUNTS))
    parser.add_argument('--config'); parser.add_argument('--candidates'); parser.add_argument('--freeze')
    parser.add_argument('--audit'); parser.add_argument('--v1-provenance')
    parser.add_argument('--dataset'); parser.add_argument('--scores'); parser.add_argument('--output', required=True)
    parser.add_argument('--development-report'); parser.add_argument('--gate'); parser.add_argument('--preflight')
    parser.add_argument('--prior-dataset', action='append', default=[])
    parser.add_argument('--local-files-only', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.resume and args.command != 'score':
        parser.error('--resume applies only to scoring')
    if args.command == 'reanalyze':
        if not args.dataset or not args.scores:
            parser.error('--dataset and --scores required')
        fresh_bundle(args.output)
        from src.cross_model.reanalysis import markdown_report, reanalyze
        if Path(args.output + '.md').exists():
            raise FileExistsError(args.output + '.md')
        result = reanalyze(args.dataset, args.scores)
        write_new(args.output, sealed(result))
        with Path(args.output + '.md').open('x', encoding='utf8') as handle:
            handle.write(markdown_report(result))
        return
    if args.command in ('preview', 'audit') or (args.command == 'generate' and not args.config):
        if not args.stage:
            parser.error('--stage is required')
        if args.stage == 'confirmatory':
            raise ValueError('confirmation preview generation is blocked until a passing recomputed gate; use generate with config/candidates/gate/preflight')
        fresh_bundle(args.output)
        priors = prior_paths(args)
        report = prompt_audit(args.stage, priors)
        if args.command == 'audit':
            write_new(args.output, sealed(report))
        else:
            histories, rows = data.generate(args.stage, priors)
            write_new(args.output, sealed({**report, 'histories': histories, 'rows': rows, 'dataset_sha256': digest(rows)}))
        print(json.dumps({'status': report['status'], 'n_members': report['n_members'], 'sample_pairs': len(report['sample_pairs'])}))
        return
    if args.command == 'freeze':
        fresh_bundle(args.output)
        if not args.audit:
            parser.error('--audit is required before freezing the corrected design')
        audit = read_sealed(args.audit)
        if (audit.get('contract') != design.CONTRACT or audit.get('code_sha256') != design.code_hash()
                or audit.get('stage') != 'development'):
            raise ValueError('reviewed development prompt audit differs from the corrected design')
        configs = {}
        for slug in design.MODEL_SETTINGS:
            path = f'configs/cross_model_relational_v2/{slug}.yaml'
            config = load_config(path)
            configs[slug] = {'path': path, 'sha256': sha256_file(path), 'config': config,
                             'original_v1_lineage': design.verify_v1_lineage(config)}
        write_new(args.output, sealed({'contract': design.CONTRACT, 'code_sha256': design.code_hash(),
                  'configs': configs, 'prompt_audit_path': str(Path(args.audit).resolve()),
                  'prompt_audit_sha256': sha256_file(args.audit),
                  'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                  'status': 'corrected design frozen; no model inference authorized or executed by this artifact',
                  'provenance': provenance({}, None)}))
        return
    if not args.config:
        parser.error('--config is required')
    config = load_config(args.config)
    design.validate_config(config)
    if args.command == 'validate':
        fresh_bundle(args.output)
        if not args.freeze:
            parser.error('--freeze is required for Stage 0')
        frozen = design.verify_freeze(args.freeze)
        if sha256_file(args.config) != frozen['configs'][design.model_slug(config)]['sha256']:
            raise ValueError('config differs from frozen model configuration')
        continuity = design.verify_v1_lineage(config, args.v1_provenance)
        tok = tokenizer(config, args.local_files_only)
        _, rows = data.generate('validation')
        result = token_audit(tok, rows)
        for key in ('chat_template_sha256', 'tokenizer_sha256'):
            if result[key] != continuity[key]:
                raise ValueError(f'actual tokenizer continuity with v1 differs: {key}')
        if digest(result['events']) != continuity['events_sha256']:
            raise ValueError('surface event universe differs from the actual frozen v1 candidate map')
        from transformers import AutoConfig
        result.update(contract=design.CONTRACT, stage='validation', validation_seed=data.SEEDS['validation'],
                      provenance=design.manifest(config, args.config), original_v1_lineage=continuity,
                      freeze_path=str(Path(args.freeze).resolve()), freeze_sha256=sha256_file(args.freeze),
                      model_config=AutoConfig.from_pretrained(config['model']['id'], revision=config['model']['revision'],
                         trust_remote_code=config['model'].get('trust_remote_code', False), local_files_only=args.local_files_only).to_dict())
        write_new(args.output, sealed(result))
        return
    candidate = load_candidate(args, config)
    if args.command == 'generate':
        generate_stage(args, config, candidate)
    elif args.command in ('score', 'plan'):
        if not args.dataset:
            parser.error('--dataset is required')
        if args.command == 'score':
            score_stage(args, config, candidate)
        else:
            rows, info = dataset_info(args.dataset, config, args.config, args.candidates, design=design)
            tok = tokenizer(config, args.local_files_only)
            check_tokenizer(tok, candidate)
            prompts = {data.render(r, tok, True) for r in rows}
            forwards = 1 + sum(len(e['ids']) > 1 for members in candidate['events'].values() for e in members)
            write_new(args.output, sealed({'stage': info['stage'], 'n_members': len(rows), 'n_unique_prompts': len(prompts),
                      'forward_upper_bound': len(prompts)*forwards, 'no_inference': True,
                      'provenance': design.manifest(config, args.config, args.candidates, args.dataset)}))
    elif args.command == 'analyze':
        if not args.dataset or not args.scores or not args.stage:
            parser.error('--dataset, --scores and --stage required')
        fresh_bundle(args.output)
        if args.stage == 'frozen_gate':
            if not args.development_report:
                parser.error('--development-report required for gate analysis')
            result = gate_report(config, args.config, args.candidates, args.dataset, args.scores,
                                 args.development_report, design=design)
        else:
            rows, scores, _, _ = score_info(args.scores, args.dataset, config, args.config,
                                           args.candidates, args.stage, design=design)
            result = {'stage': args.stage, 'dataset_path': str(Path(args.dataset).resolve()),
                      'scores_path': str(Path(args.scores).resolve()), 'dataset_sha256': sha256_file(args.dataset),
                      'scores_sha256': sha256_file(args.scores),
                      'provenance': design.manifest(config, args.config, args.candidates, args.dataset)}
            if args.stage == 'development':
                result.update(evaluation=design.evaluate(rows, scores, gate=False), eligibility_decision='descriptive_only')
            else:
                verify_confirmation(args.dataset, config, args.config, args.candidates, design=design)
                if any(s['score_kind'] != 'confirmatory' for s in scores):
                    raise ValueError('confirmation analysis requires confirmatory scoring mode')
                from src.cross_model.robustness_analysis import report
                result.update(report(rows, scores))
        write_new(args.output, sealed(result))
    elif args.command == 'preflight':
        if not args.gate:
            parser.error('--gate required')
        gate = verify_gate(args.gate, config, args.config, args.candidates, design=design)
        dev = read_sealed(local_artifact_path(gate['development_report_path']))
        write_new(args.output, sealed({'stage': 'preflight', 'contract': design.CONTRACT,
                  'gate_sha256': sha256_file(args.gate), 'frozen_gate_competence': gate['evaluation'],
                  'development_competence': dev['evaluation'], 'expected_cells': 384,
                  'surface_geometry_audit': candidate['surface_geometry_audit'],
                  'freeze_sha256': candidate['freeze_sha256'],
                  'confirmatory_history_count': 96, 'all_trial_primary': True,
                  'provenance': design.manifest(config, args.config, args.candidates, local_artifact_path(gate['dataset_path']))}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never overwrite a failed attempt or reinterpret a technical stop as competence.
        import sys
        if '--output' in sys.argv:
            output = sys.argv[sys.argv.index('--output')+1]
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            write_new(output + f'.stopped-{stamp}.json', {'status': 'stopped', 'error': type(error).__name__,
                      'reason': str(error), 'argv': sys.argv[1:], 'timestamp_utc': stamp,
                      'provenance': provenance({}, None)})
        raise
