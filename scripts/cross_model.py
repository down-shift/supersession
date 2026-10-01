#!/usr/bin/env python3
"""Strict cross_model_v1 stages; default development/gate scoring never computes R."""
import argparse
import json
import logging
from pathlib import Path

from src.cross_model.protocol import (CONTRACT, COUNTS, GATE, VALUES, VERSION, check_manifest,
    code_hash, digest, disjoint, evaluate, manifest, read_sealed, sealed, validate_config,
    validate_dataset, write_new)
from src.cross_model.dataset import generate
from src.cross_model.tokens import audit_pairs, check_tokenizer, validate, surface_geometry_audit
from src.cross_model.workflow import (dataset_info, gate_report, local_artifact_path,
                                      score_info, verify_confirmation, verify_gate)
from src.data.io import read_jsonl, sha256_file
from src.utils import load_config
from src.cross_model.progress import configure_logging, progress

logger = logging.getLogger("cross_model")


def load_candidate(a, c):
    candidate = read_sealed(a.candidates)
    check_manifest(candidate['provenance'], c, a.config)
    if candidate.get('contract') != CONTRACT: raise ValueError('candidate contract mismatch')
    if candidate.get('surface_geometry_audit') != surface_geometry_audit(candidate['events']):
        raise ValueError('mandatory surface geometry audit missing or inconsistent')
    return candidate


def tokenizer(c, local=False):
    from transformers import AutoTokenizer
    m = c['model']
    return AutoTokenizer.from_pretrained(m.get('tokenizer_id') or m['id'], revision=m['tokenizer_revision'],
        trust_remote_code=m.get('trust_remote_code', False), local_files_only=local)


def fresh_bundle(output):
    for suffix in ('', '.run.json', '.provenance.json', '.analysis.json'):
        if Path(str(output)+suffix).exists(): raise FileExistsError(str(output)+suffix)


def write_dataset(output, rows, info):
    fresh_bundle(output)
    p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        for row in progress(rows, desc='Writing dataset', unit='row'):
            f.write(json.dumps(row, sort_keys=True)+'\n')
    return info


def claim_stage(c, stage, output):
    """One dataset per model/stage even when fresh output paths are used."""
    key = digest({'model':c['model']['id'], 'revision':c['model']['revision'],
                  'contract_sha256':digest(CONTRACT)})[:16]
    path = Path('outputs/cross_model_v1/stage_claims')/key/(stage+'.json')
    write_new(path, {'stage':stage,'dataset_path':str(Path(output).resolve()),
                     'model':c['model'], 'contract_sha256':digest(CONTRACT)})


def main():
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['validate', 'generate', 'score', 'analyze', 'preflight', 'mechanism', 'sensitivity', 'smoke', 'mechanism-plan'])
    p.add_argument('--config', required=True); p.add_argument('--candidates')
    p.add_argument('--stage', choices=list(COUNTS)); p.add_argument('--dataset'); p.add_argument('--scores')
    p.add_argument('--output', required=True); p.add_argument('--development-report')
    p.add_argument('--gate'); p.add_argument('--preflight'); p.add_argument('--resume', action='store_true')
    p.add_argument('--prior-dataset', action='append', default=[])
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--heads', action='store_true', help='optional fixed discovery/heldout M5 battery')
    a = p.parse_args(); c = load_config(a.config); validate_config(c)
    logger.info("Starting command=%s config=%s output=%s", a.command, a.config, a.output)
    if a.command == 'validate':
        fresh_bundle(a.output)
        logger.info("Loading pinned tokenizer and beginning tokenizer-only Stage 0 audit")
        tok = tokenizer(c, a.local_files_only)
        rows = generate('validation', 4)
        result = validate(tok, rows)
        result.update(stage='validation', validation_seed=CONTRACT['seeds']['validation'], contract=CONTRACT, provenance=manifest(c, a.config),
                      selection_provenance='fixed ordinary values; tokenizer/format rules only; no logits',
                      model_config=__import__('transformers').AutoConfig.from_pretrained(
                          c['model']['id'], revision=c['model']['revision'],
                          trust_remote_code=c['model'].get('trust_remote_code', False),
                          local_files_only=a.local_files_only).to_dict())
        write_new(a.output, sealed(result))
        logger.info("Stage 0 complete: %d prefixes, %d edit pairs; artifact=%s",
                    result['unique_prefixes_checked'], result['edit_audit']['pairs_checked'], a.output)
        print(json.dumps({'status': 'passed', 'raw_R_defined': result['canonical_raw_R_defined'],
                          'aligned_pairs': result['edit_audit']['all_mechanism_aligned'],
                          'surface_geometry_audit': result['surface_geometry_audit']})); return
    if not a.candidates: p.error('--candidates required')
    candidate = load_candidate(a, c)
    if a.command == 'smoke':
        fresh_bundle(a.output)
        from src.cross_model.runtime import load_pinned_model, hook_smoke
        from src.data.supersession_behavior import render_behavior_example
        logger.info("Loading model for real hook smoke")
        model,tok = load_pinned_model(c);check_tokenizer(tok,candidate)
        result = hook_smoke(model,tok,render_behavior_example(generate('validation',4)[0],tok,True))
        write_new(a.output,sealed({**result,'provenance':manifest(c,a.config,a.candidates)}))
        logger.info("Hook smoke %s; artifact=%s", result['status'], a.output); return
    if a.command == 'generate':
        if not a.stage: p.error('--stage required')
        fresh_bundle(a.output)
        info = {'stage': a.stage}
        prior = list(a.prior_dataset)
        legacy = ['mistral_nl_dev.jsonl','mistral_nl_frozen_gate.jsonl',
                  'phi4_mini_nl_dev.jsonl','phi4_mini_nl_frozen_gate.jsonl',
                  'nl_dev.jsonl','nl_frozen_gate.jsonl','nl_confirmatory.jsonl']
        prior += [str(Path('outputs/supersession')/x) for x in legacy
                  if (Path('outputs/supersession')/x).exists()]
        # Prevent replaying histories from an earlier run of this same model.
        model_slug = Path(a.config).stem
        for root in (Path('outputs')/model_slug, Path('outputs/cross_model_v1')/model_slug):
            prior += [str(root/name) for name in ('development.jsonl','gate.jsonl','confirmatory.jsonl')
                      if (root/name).exists() and Path(root/name).resolve() != Path(a.output).resolve()]
        prior = list(dict.fromkeys(prior))
        if a.stage == 'frozen_gate':
            if not a.development_report: p.error('--development-report required')
            dev = read_sealed(a.development_report)
            dr, ds, _, _ = score_info(dev['scores_path'], dev['dataset_path'], c, a.config, a.candidates, 'development')
            if evaluate(dr, ds, gate=False) != dev['evaluation']: raise ValueError('development report mismatch')
            # Fixed prompt, vocabulary and format: no per-model redevelopment in v1.
            info.update(development_report_sha256=sha256_file(a.development_report))
            prior.append(dev['dataset_path'])
        if a.stage == 'confirmatory':
            if not a.gate or not a.preflight: p.error('--gate and --preflight required before confirmation')
            gate = verify_gate(a.gate, c, a.config, a.candidates)
            report = read_sealed(a.preflight)
            if report['gate_sha256'] != sha256_file(a.gate): raise ValueError('wrong preflight gate')
            check_manifest(report['provenance'], c, a.config, a.candidates,
                           local_artifact_path(gate['dataset_path']))
            info.update(gate_path=str(Path(a.gate).resolve()), gate_sha256=sha256_file(a.gate),
                        preflight_path=str(Path(a.preflight).resolve()), preflight_sha256=sha256_file(a.preflight))
            dev = read_sealed(local_artifact_path(gate['development_report_path']))
            prior += [str(local_artifact_path(gate['dataset_path'])),
                      str(local_artifact_path(dev['dataset_path']))]
        logger.info("Generating %s histories (n=%d)", a.stage, COUNTS[a.stage])
        rows = generate(a.stage, COUNTS[a.stage]); validate_dataset(rows, a.stage)
        info.update(history_signatures=disjoint(rows, prior), prior_datasets=[
            {'path': str(Path(x).resolve()), 'sha256': sha256_file(x)} for x in prior])
        logger.info("Auditing tokenizer, prompt edits, and history overlap")
        tok = tokenizer(c, a.local_files_only); check_tokenizer(tok, candidate)
        audit = validate(tok, rows)
        if audit['events'] != candidate['events']: raise ValueError('actual dataset surface map mismatch')
        info.update(edit_audit=audit['edit_audit'], tokenizer_sha256=candidate['tokenizer_sha256'],
                    chat_template_sha256=candidate['chat_template_sha256'])
        claim_stage(c, a.stage, a.output)
        write_dataset(a.output, rows, info)
        info['provenance'] = manifest(c, a.config, a.candidates, a.output)
        write_new(a.output+'.provenance.json', sealed(info)); print(f'wrote {len(rows)} rows')
        logger.info("Generation complete: rows=%d histories=%d artifact=%s", len(rows), len({r['history_id'] for r in rows}), a.output); return
    if a.command == 'score':
        if not a.dataset: p.error('--dataset required')
        rows, info = dataset_info(a.dataset, c, a.config, a.candidates)
        if info['stage'] == 'confirmatory':
            verify_confirmation(a.dataset, c, a.config, a.candidates)
            gate = read_sealed(info['gate_path'])
            runtime = read_sealed(str(local_artifact_path(gate['scores_path']))+'.provenance.json')['provenance']
            current = manifest(c,a.config,a.candidates,a.dataset)
            if any(runtime[k] != current[k] for k in ('packages','python')):
                raise ValueError('confirmatory runtime differs from frozen gate')
        if not a.resume: fresh_bundle(a.output)
        if Path(a.output+'.provenance.json').exists():
            if not a.resume: raise FileExistsError(a.output)
            score_info(a.output, a.dataset, c, a.config, a.candidates, info['stage'])
            print('already complete'); return
        from src.cross_model.runtime import load_pinned_model
        from src.data.progress import prepare_jsonl_progress, append_jsonl_record
        from src.cross_model.scoring import score_row
        logger.info("Loading model/tokenizer for %s scoring; checkpoint output=%s", info['stage'], a.output)
        model, tok = load_pinned_model(c); check_tokenizer(tok, candidate)
        actual_map = validate(tok, rows)
        if actual_map['events'] != candidate['events']: raise ValueError('loaded continuation map mismatch')
        prov = manifest(c, a.config, a.candidates, a.dataset)
        prov.update(model_config_sha256=digest(json.loads(model.config.to_json_string())),
                    chat_template_sha256=candidate['chat_template_sha256'], tokenizer_sha256=candidate['tokenizer_sha256'],
                    dataset_seed=sorted({r['seed'] for r in rows}), prompt_edit_audit=actual_map['edit_audit'],
                    resolved_device_map={k:str(v) for k,v in getattr(model, 'hf_device_map', {}).items()})
        fingerprint = {k: v for k, v in prov.items() if k != 'timestamp_utc'}
        completed = prepare_jsonl_progress(a.output, a.dataset, a.candidates, fingerprint, rows, resume=a.resume)
        saved = read_jsonl(a.output) if Path(a.output).exists() else []
        expected = {r['example_id']: r for r in rows}
        from src.data.supersession_behavior import render_behavior_example
        for s in saved:
            r = expected[s['example_id']]
            if any(s.get(k) != v for k,v in r.items()) or s['prompt'] != render_behavior_example(r, tok, True):
                raise ValueError('resume score metadata or prompt differs')
        cache = {}
        scored_rows = 0
        pending_prompts = {render_behavior_example(r, tok, True) for r in rows
                           if r['example_id'] not in completed}
        forwards_per_prompt = 1 + sum(len(event['ids']) > 1
                                      for members in candidate['events'].values() for event in members)
        total_forwards = len(pending_prompts) * forwards_per_prompt
        logger.info("Scoring %d examples (%d prompts, up to %d model forwards; prior rows=%d)",
                    len(rows), len(pending_prompts), total_forwards, len(completed))
        with progress(total=total_forwards, desc='Model forwards', unit='forward') as forward_bar:
            for r in progress(rows, desc=f"Scoring {info['stage']}", unit='row', leave=False):
                if r['example_id'] in completed: continue
                prompt = render_behavior_example(r, tok, True)
                if prompt not in cache:
                    cache[prompt] = score_row(model, tok, r, candidate,
                                              info['stage'] != 'confirmatory',
                                              progress_callback=forward_bar.update)
                append_jsonl_record(a.output, {**cache[prompt], **r})
                scored_rows += 1
        write_new(a.output+'.provenance.json', sealed({'stage': info['stage'], 'scores_sha256':sha256_file(a.output),
                                                     'provenance':prov}))
        print(f'scored {scored_rows} members; causal effects not computed by scorer')
        logger.info("Scoring complete: rows=%d artifact=%s", scored_rows, a.output); return
    if a.command == 'analyze':
        if not a.dataset or not a.scores or not a.stage: p.error('--dataset, --scores, --stage required')
        fresh_bundle(a.output)
        if a.stage == 'frozen_gate':
            if not a.development_report: p.error('--development-report required')
            result = gate_report(c, a.config, a.candidates, a.dataset, a.scores, a.development_report)
        else:
            rows, scores, _, _ = score_info(a.scores, a.dataset, c, a.config, a.candidates, a.stage)
            if a.stage == 'development':
                evaluation = evaluate(rows, scores, gate=False)
                result = {'stage': 'development', 'evaluation':evaluation,
                    'dataset_path':str(Path(a.dataset).resolve()), 'scores_path':str(Path(a.scores).resolve()),
                    'dataset_sha256':sha256_file(a.dataset), 'scores_sha256':sha256_file(a.scores),
                    'stop_reason': None, 'eligibility_decision': 'descriptive_only'}
            else:
                verify_confirmation(a.dataset, c, a.config, a.candidates)
                if any(s['score_kind'] != 'confirmatory' for s in scores): raise ValueError('wrong scoring mode')
                from src.cross_model.analysis import contrasts, summarize, strata
                histories = contrasts(rows, scores)
                result = {'stage':'confirmatory', 'sequence_mass':summarize(histories),
                          'history_rows':histories, 'strata':strata(rows, histories),
                          'estimand_units':'log probability odds (nats), not original single-token logits',
                          'original_raw_logit': summarize(contrasts(rows, scores, 'canonical_candidate_logits'))
                             if candidate['canonical_raw_R_defined'] else {'available':False, 'reason':'multi-token canonical values'}}
            result['provenance'] = manifest(c, a.config, a.candidates, a.dataset)
        write_new(a.output, sealed(result)); print(json.dumps({'stage':result['stage'], 'pass':result.get('evaluation',{}).get('pass')}))
        logger.info("Analysis complete: stage=%s artifact=%s", result['stage'], a.output); return
    if a.command == 'preflight':
        if not a.gate: p.error('--gate required')
        gate = verify_gate(a.gate, c, a.config, a.candidates)
        dev = read_sealed(local_artifact_path(gate['development_report_path']))
        report = {'stage':'preflight', 'gate_sha256':sha256_file(a.gate), 'model':c['model'],
                  'vocabulary':VALUES, 'surface_policy':CONTRACT['surfaces'],
                  'surface_geometry_audit': candidate['surface_geometry_audit'],
                  'development_competence':dev['evaluation'], 'frozen_gate_criteria':GATE,
                  'frozen_gate_competence':gate['evaluation'], 'expected_cells':64,
                  'confirmatory_estimands':{'primary':'R_superseded - R_irrelevant_counterbalanced',
                    'secondary':'R_live - R_superseded', 'score':'bounded surface-class continuation mass (prefix events, no termination)',
                    'original_raw_R_defined':candidate['canonical_raw_R_defined']},
                  'mechanistic_hooks':['block_output','attention_output (post projection)','mlp_output'],
                  'head_hooks':'pre-o_proj; runtime dimension and hook checks required',
                  'caveats':{'variable_length_edit':not candidate['edit_audit']['all_mechanism_aligned'],
                             'remote_code':c['model'].get('trust_remote_code',False),
                             'real_model_hook_validation':'tiny native models passed; pinned remote Phi and quantized GPU execution pending'
                             , 'raw_logit_comparability':'available only when all canonical values are one-token; do not compare to sequence masses'},
                  'provenance':manifest(c, a.config, a.candidates,
                                         local_artifact_path(gate['dataset_path']))}
        write_new(a.output, sealed(report)); print(json.dumps(report, indent=2)); return
    if a.command == 'mechanism-plan':
        if not a.dataset: p.error('--dataset required')
        if a.heads: p.error('mechanism-plan describes the core battery')
        fresh_bundle(a.output)
        rows,_ = verify_confirmation(a.dataset,c,a.config,a.candidates)
        tok = tokenizer(c,a.local_files_only);check_tokenizer(tok,candidate)
        from src.cross_model.mechanism import execution_plan
        result = execution_plan(rows,tok,candidate,candidate['model_config']['num_hidden_layers'])
        write_new(a.output,sealed({'stage':'mechanism_plan',**result,
            'provenance':manifest(c,a.config,a.candidates,a.dataset)}))
        print(json.dumps(result,indent=2)); return
    if a.command == 'sensitivity':
        if not a.dataset or not a.scores: p.error('--dataset and --scores required')
        fresh_bundle(a.output)
        from src.cross_model.sensitivity import run
        run(a,c,candidate); return
    if a.command == 'mechanism':
        if not a.dataset or not a.scores: p.error('--dataset and --scores required')
        if not a.resume: fresh_bundle(a.output)
        if a.resume and Path(a.output+'.provenance.json').exists():
            verify_confirmation(a.dataset,c,a.config,a.candidates)
            score_info(a.scores,a.dataset,c,a.config,a.candidates,'confirmatory')
            done = read_sealed(a.output+'.provenance.json')
            check_manifest(done['provenance'],c,a.config,a.candidates,a.dataset)
            if done['scores_sha256'] != sha256_file(a.output): raise ValueError('completed mechanism hash mismatch')
            if done['provenance']['mode'] != ('heads' if a.heads else 'core'): raise ValueError('mechanism resume mode mismatch')
            if done['provenance']['score_sha256'] != sha256_file(a.scores): raise ValueError('behavioral score binding changed')
            read_sealed(a.output+'.analysis.json')
            print('mechanistic run already complete'); return
        from src.cross_model.mechanism import run
        run(a, c, candidate)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import sys
        from datetime import datetime, timezone
        logging.getLogger('cross_model').error("Command failed: %s: %s", type(error).__name__, error)
        args = sys.argv[1:]
        if '--output' in args:
            output = args[args.index('--output')+1]
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            write_new(output+'.stopped-'+stamp+'.json', sealed({'stage':'technical_or_protocol_stop',
                      'command':args,'reason':str(error),'error_type':type(error).__name__,
                      'code_sha256':code_hash()}))
        raise
