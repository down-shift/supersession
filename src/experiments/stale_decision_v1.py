"""Terminated scoring, tokenizer audits and strict checkpoint contracts."""
import json
import hashlib
import math
import os
import re
from pathlib import Path
from src.data.stale_decision_v1 import build_prompt, digest, read_rows, validate, write_new
from src.cross_model.tokens import encode, continuations, token_span
from src.cross_model.scoring import score_prompt

ROOT = Path(__file__).resolve().parents[2]


def code_hash():
    files = ['src/data/stale_decision_v1.py', 'src/experiments/stale_decision_v1.py',
             'src/analysis/stale_decision_v1.py', 'scripts/stale_decision_v1.py',
             'src/cross_model/scoring.py', 'src/cross_model/tokens.py', 'docs/stale_decision_v1.md']
    return digest({p: (ROOT / p).read_text() for p in files})


def check_config(config):
    if config.get('protocol') != 'stale_decision_v1':
        raise ValueError('wrong protocol')
    for key in ('revision', 'tokenizer_revision'):
        if not re.fullmatch('[a-f0-9]{40}', str(config['model'].get(key, ''))):
            raise ValueError('immutable model/tokenizer revisions required')
    if any(config.get(k) != v for k, v in [('development_replicates', 1), ('gate_replicates', 2), ('confirmation_replicates', 8)]):
        raise ValueError('frozen split counts differ')
    if config.get('max_new_tokens') != 16 or config.get('bootstrap_draws') != 2000:
        raise ValueError('frozen decoding/bootstrap settings differ')


def rendered(tokenizer, body):
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError('actual tokenizer requires chat template')
    # Body ends with Answer: for stable plain prefix; chat generation prefix is authoritative.
    return tokenizer.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def action_events(tokenizer, prompt, actions):
    """Full label plus one designated EOS event, not unbounded prefix mass."""
    prefix = encode(tokenizer, prompt)
    eos = tokenizer.eos_token_id
    if not prefix or not isinstance(eos, int):
        raise ValueError('nonempty prefix and designated EOS required')
    events, sequences = {}, []
    for action in actions:
        text = ' ' + action
        full = encode(tokenizer, prompt + text)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError('candidate retokenizes prefix')
        ids = full[len(prefix):] + [eos]
        if len(ids) > 16:
            raise ValueError('canonical action cannot finish within fixed generation budget')
        if eos in ids[:-1]:
            raise ValueError('EOS inside candidate')
        if any(ids[:len(s)] == s or s[:len(ids)] == ids for s in sequences):
            raise ValueError('candidate continuation collision/prefix overlap')
        sequences.append(ids)
        events[action] = [{'text': text, 'ids': ids}]
    return events


def tokenizer_fingerprint(tok):
    return digest({'backend': tok.backend_tokenizer.to_str(), 'chat_template': tok.chat_template,
                   'eos': tok.eos_token_id, 'special_tokens': tok.special_tokens_map})


def token_audit(rows, tok):
    validate(rows)
    result = {'dataset_hash': digest(rows), 'tokenizer_hash': tokenizer_fingerprint(tok),
              'prompts': {}, 'pairs': [], 'score_event': 'space + full action label + designated EOS',
              'termination_scope': 'one canonical EOS; excludes other whitespace and termination paths'}
    pair_groups = {}
    for r in rows:
        eid = r['example_id']; variants = {}
        for name, query, other in [('downstream', 'downstream', False), ('easy_target', 'easy', False), ('easy_other', 'easy', True)]:
            body, char_span = build_prompt(r, query, other)
            prompt = rendered(tok, body)
            events = action_events(tok, prompt, r['actions']) if query == 'downstream' else continuations(tok, prompt, values=list(r['policy']))
            if len(encode(tok, prompt)) + 16 > getattr(tok, 'model_max_length', float('inf')):
                raise ValueError('prompt exceeds tokenizer context budget')
            offsets = None
            if char_span:
                # Find the complete body in the exact chat-rendered prefix.
                if prompt.count(body) != 1:
                    raise ValueError('chat template must preserve body exactly once')
                shift = prompt.index(body)
                offsets = token_span(tok, prompt, shift + char_span[0], shift + char_span[1])
            variants[name] = {'prompt_hash': digest(prompt), 'n_tokens': len(encode(tok, prompt)),
                              'edit_token_span': offsets, 'events': events}
        result['prompts'][eid] = variants
        pair_groups.setdefault(r['pair_id'], {})[r['member']] = r
    for pid, members in pair_groups.items():
        if len(members) == 1:
            continue
        a, b = members[0], members[1]
        for name, query, other in [('downstream', 'downstream', False), ('easy_target', 'easy', False), ('easy_other', 'easy', True)]:
            pa = rendered(tok, build_prompt(a, query, other)[0]); pb = rendered(tok, build_prompt(b, query, other)[0])
            ai, bi = encode(tok, pa), encode(tok, pb)
            aa = result['prompts'][a['example_id']][name]['edit_token_span']
            bb = result['prompts'][b['example_id']][name]['edit_token_span']
            if ai[:aa[0]] != bi[:bb[0]] or ai[aa[-1]+1:] != bi[bb[-1]+1:] or ai[aa[0]:aa[-1]+1] == bi[bb[0]:bb[-1]+1]:
                raise ValueError('token edit alters surrounding context or has no change')
            result['pairs'].append({'pair_id': pid, 'query': name, 'baseline_span': aa, 'edit_span': bb,
                                    'baseline_ids': ai[aa[0]:aa[-1]+1], 'edit_ids': bi[bb[0]:bb[-1]+1],
                                    'length_delta': len(bi)-len(ai)})
    result['token_counts_by_family'] = {}
    for family in sorted({r['family'] for r in rows}):
        counts = [result['prompts'][r['example_id']]['downstream']['n_tokens'] for r in rows if r['family'] == family]
        result['token_counts_by_family'][family] = {'min': min(counts), 'max': max(counts), 'mean': sum(counts)/len(counts)}
    return result


def classify(text, actions, correct, terminated=True):
    # Only surrounding whitespace is normalized; formatting/explanations remain invalid.
    exact = text.strip()
    strict = ('correct' if exact == correct else 'wrong' if exact in actions else 'invalid') if terminated else 'invalid'
    hits = [a for a in actions if re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', text)]
    return {'classification': strict, 'relaxed_action': hits[0] if len(hits) == 1 else None}


def checked_scores(rows, scores, complete=True):
    validate(rows)
    expected = {r['example_id']: r for r in rows}; seen = set(); models = set()
    for s in scores:
        eid = s['example_id']
        if eid in seen or eid not in expected or s.get('row_hash') != digest(expected[eid]):
            raise ValueError('duplicate/unknown/stale score')
        seen.add(eid); models.add(s['model_id'])
        r = expected[eid]
        for field, labels in [('action_logp', r['actions']), ('easy_target', r['policy']), ('easy_other', r['policy'])]:
            if set(s[field]) != set(labels) or not all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 0 for v in s[field].values()):
                raise ValueError('missing/nonfinite/invalid scores')
        if any(sum(math.exp(v) for v in s[field].values()) > 1 + 1e-6 for field in ('easy_target', 'easy_other')):
            raise ValueError('retrieval disjoint event mass exceeds one')
        if s['generated'] != classify(s['raw_text'], r['actions'], r['answer'], s['terminated']):
            raise ValueError('generation parse mismatch')
        if s['easy_generated'] != classify(s['easy_raw_text'], list(r['policy']), r['correct_state'], s['easy_terminated']):
            raise ValueError('easy generation parse mismatch')
        mass = sum(math.exp(v) for v in s['action_logp'].values())
        if mass > 1 + 1e-6 or abs(s['canonical_valid_mass'] - mass) > 1e-9:
            raise ValueError('invalid canonical probability mass')
        if not all(math.isfinite(s[k]) for k in ('prompt_tokens', 'history_position')):
            raise ValueError('nonfinite predictors')
    if len(models) > 1:
        raise ValueError('analyze models separately')
    if complete and seen != set(expected):
        raise ValueError('missing score records')
    return seen


def run(rows, config, output, model, tok, resume=False):
    check_config(config)
    audit = token_audit(rows, tok)  # Actual loaded tokenizer, before any forward.
    manifest = {'protocol': 'stale_decision_v1', 'dataset_hash': digest(rows), 'config_hash': digest(config),
                'code_hash': code_hash(), 'tokenizer_hash': audit['tokenizer_hash'], 'audit_hash': digest(audit),
                'config': config, 'runtime': runtime_info(model)}
    output = Path(output); sidecar = str(output) + '.run.json'
    if resume:
        if json.loads(Path(sidecar).read_text()) != manifest:
            raise ValueError('resume manifest mismatch')
        saved = recover_jsonl(output)
        done = checked_scores(rows, saved, complete=False)
    else:
        if output.exists() or Path(sidecar).exists() or Path(str(output)+'.audit.json').exists() or Path(str(output)+'.easy.jsonl').exists():
            raise FileExistsError(output)
        write_new(sidecar, manifest); write_new(str(output)+'.audit.json', audit)
        output.touch(exist_ok=False)
        done = set()
    easy_path = Path(str(output) + '.easy.jsonl')
    easy_saved = recover_jsonl(easy_path) if resume else []
    easy_cache = validate_easy(rows, easy_saved)
    for r in rows:
        eid = r['example_id']
        if eid in easy_cache:
            continue
        values = {}
        for name, other in [('easy_target', False), ('easy_other', True)]:
            p = rendered(tok, build_prompt(r, 'easy', other)[0])
            events = continuations(tok, p, values=list(r['policy']))
            values[name] = score_prompt(model, tok, p, events)[0]
            if name == 'easy_target':
                easy_raw, easy_ids, easy_terminated = generate_response(model, tok, p, config['max_new_tokens'])
        saved_easy = dict(example_id=eid, row_hash=digest(r), **values, easy_raw_text=easy_raw,
                          easy_token_ids=easy_ids, easy_terminated=easy_terminated,
                          easy_generated=classify(easy_raw, list(r['policy']), r['correct_state'], easy_terminated))
        validate_easy(rows, [saved_easy])
        append_record(easy_path, saved_easy); easy_cache[eid] = saved_easy
    if resume:
        for saved_score in saved:
            original_easy = easy_cache[saved_score['example_id']]
            if any(saved_score[k] != original_easy[k] for k in ('easy_target', 'easy_other', 'easy_raw_text', 'easy_generated')):
                raise ValueError('downstream record differs from easy checkpoint')
    # All easy retrieval interventions finish before the first downstream forward.
    for r in rows:
        if r['example_id'] in done:
            continue
        easy = easy_cache[r['example_id']]
        values = {k: easy[k] for k in ('easy_target', 'easy_other')}
        easy_raw, easy_ids, easy_terminated = easy['easy_raw_text'], easy['easy_token_ids'], easy['easy_terminated']
        p = rendered(tok, build_prompt(r)[0])
        values['action_logp'] = score_prompt(model, tok, p, action_events(tok, p, r['actions']))[0]
        raw, raw_ids, terminated = generate_response(model, tok, p, config['max_new_tokens'])
        s = dict(example_id=r['example_id'], row_hash=digest(r), model_id=config['model']['id'], **values,
                 raw_text=raw, generated_token_ids=raw_ids, terminated=terminated, easy_raw_text=easy_raw, easy_token_ids=easy_ids, easy_terminated=easy_terminated,
                 generated=classify(raw, r['actions'], r['answer'], terminated),
                 easy_generated=classify(easy_raw, list(r['policy']), r['correct_state'], easy_terminated),
                 canonical_valid_mass=sum(math.exp(v) for v in values['action_logp'].values()),
                 prompt_tokens=audit['prompts'][r['example_id']]['downstream']['n_tokens'],
                 history_position=(audit['prompts'][r['example_id']]['downstream']['edit_token_span'] or [-1])[0])
        checked_scores(rows, [s], complete=False)
        append_record(output, s)
    checked_scores(rows, read_rows(output))


def runtime_info(model):
    import platform
    import importlib.metadata
    return {'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy')},
            'model_class': type(model).__name__, 'model_config_hash': digest(model.config.to_dict()),
            'device_map': {k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
            'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()})}


def generate_response(model, tok, prompt, limit):
    import torch
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([encode(tok, prompt)], device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, do_sample=False, num_beams=1, max_new_tokens=limit,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                             return_dict_in_generate=True, output_scores=True)
    if any(not torch.isfinite(s).all() for s in out.scores):
        raise ValueError('nonfinite generation scores')
    suffix = out.sequences[0, ids.shape[1]:].tolist()
    return tok.decode(suffix, skip_special_tokens=True), suffix, bool(suffix and suffix[-1] == tok.eos_token_id)


def append_record(path, record):
    with Path(path).open('a') as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def recover_jsonl(path):
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b''
    if raw and not raw.endswith(b'\n'):
        good = raw[:raw.rfind(b'\n')+1]
        recovery = str(path) + '.interrupted.' + hashlib.sha256(raw).hexdigest() + '.bin'
        with Path(recovery).open('xb') as f:
            f.write(raw)
        path.write_bytes(good)
    return read_rows(path) if path.exists() else []


def validate_easy(rows, scores):
    expected = {r['example_id']: r for r in rows}; cache = {}
    for s in scores:
        eid = s['example_id']
        if eid in cache or eid not in expected or s['row_hash'] != digest(expected[eid]):
            raise ValueError('invalid easy checkpoint metadata')
        r = expected[eid]
        for field in ('easy_target', 'easy_other'):
            if set(s[field]) != set(r['policy']) or not all(math.isfinite(v) and v <= 0 for v in s[field].values()):
                raise ValueError('invalid easy checkpoint scores')
        if s['easy_generated'] != classify(s['easy_raw_text'], list(r['policy']), r['correct_state'], s['easy_terminated']):
            raise ValueError('invalid easy checkpoint generation')
        cache[eid] = s
    return cache
